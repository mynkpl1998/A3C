from a3c.src.policies.A3C import MLP

import os
import gym
import torch
import time
import numpy as np
import torch.nn.functional as F
from scipy.signal import lfilter
from tensorboardX import SummaryWriter


def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad

def train_process(rank, args, shared_model, counter, lock, optimizer, vec_env):

    torch.manual_seed(args.getValue("torchSeed") + rank)

    # Create a parallel env and seed it
    env = vec_env.createEnv(args.getValue("normalize_state"))
    env.seed(args.getValue("seed_offset") + rank)

    # Local Copy of model
    model = MLP(vec_env.obs_size, vec_env.num_actions)

    model.train()

    state = env.reset()
    state = torch.from_numpy(state)
    done = True

    episode_length = 0

    while True:

        # Sync with the shared model
        model.load_state_dict(shared_model.state_dict())

        if done:
            hx = torch.zeros(1, 256)
        else:
            hx = hx.detach()
        
        values = []
        log_probs = []
        rewards = []
        entropies = []

        for step in range(args.getValue("num_steps")):
            
            episode_length += 1
            value, logit, hx = model((state.unsqueeze(0).float(), hx))

            prob = F.softmax(logit, dim=-1)
            log_prob = F.log_softmax(logit, dim=-1)
            entropy = -(log_prob * prob).sum(1, keepdim=True)
            entropies.append(entropy)

            action = prob.multinomial(num_samples=1).detach()
            log_prob = log_prob.gather(1, action)
            
            state, reward, done, _  = env.step(action.numpy()[0,0])
            
            with lock:
                counter.value += 1
            
            if done:
                episode_length = 0
                state = env.reset()
            
            state = torch.from_numpy(state)
            values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)

            if done:
                break
        
        R = torch.zeros(1, 1)
        if not done:
            value, _, _ = model((state.unsqueeze(0).float(), hx))
            R = value.detach()
        
        values.append(R)
        policy_loss = 0.0
        value_loss = 0.0
        gae = torch.zeros(1, 1)

        for i in reversed(range(len(rewards))):
            R = args.getValue("gamma") * R + rewards[i]
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            # Generalized Adavantage Estimation
            delta_t = rewards[i] + args.getValue("gamma") * values[i+1] - values[i]
            gae = gae * args.getValue("gamma") * args.getValue("gae_lambda") + delta_t

            policy_loss = policy_loss - log_probs[i] * gae.detach() - args.getValue("entropy_coef") * entropies[i]
        
        optimizer.zero_grad()

        (policy_loss + args.getValue("value_loss_coef") * value_loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.getValue("grad_norm"))

        ensure_shared_grads(model, shared_model)
        optimizer.step()


def buildCheckPointDict(items=[]):
    checkPointDict = {}
    for item in items:
        checkPointDict[item] = None
    return checkPointDict

def saveCheckPoint(checkPointDict, num, args):
    checkPointFile = args.getValue("log_dir") + "/" + args.getValue("exp_name") + "/saved_models/checkpoint-" + str(num)
    torch.save(checkPointDict, checkPointFile)
    print("Saved %d checkpoint !"%(num))


def test_process(rank, args, shared_model, counter, vec_env):

    # Checkpoint gap (iterations)
    checkPointGap = 500000
    currentGap = checkPointGap

    # Create Writer Object
    writer = SummaryWriter(logdir=args.getValue("log_dir")+"/"+args.getValue("exp_name")+ "/logs")

    torch.manual_seed(args.getValue("torchSeed") + rank)
    
    # Create a env in parallel and seed it
    env = vec_env.createEnv(args.getValue("normalize_state"))
    env.seed(args.getValue("seed_offset") + rank)

    # Create checkpoint dict
    checkPointDict = buildCheckPointDict(['model', 'state_dict', 'args', 'env'])
    checkPointDict["model"] = MLP(vec_env.obs_size, vec_env.num_actions)
    checkPointDict["args"] = args
    checkPointDict["env"] = vec_env
    checkPointCount = 0
    
    # Local Copy of Env
    model = MLP(vec_env.obs_size, vec_env.num_actions)
    model.eval()

    state = env.reset()
    state = torch.from_numpy(state)
    reward_sum = 0
    done = True

    start_time = time.time()

    episode_length = 0
    episode_policy_entropy = 0

    while True:

        episode_length += 1

        if done:
            model.load_state_dict(shared_model.state_dict())
            hx = torch.zeros(1, 256)
        else:
            hx = hx.detach()
        
        with torch.no_grad():
            value, logit, hx = model((state.unsqueeze(0).float(), hx))
        
        prob = F.softmax(logit, dim=-1)
        action = prob.multinomial(num_samples=1).detach()
        episode_policy_entropy += -(torch.log(prob).mul(prob).sum().item())

        #action = prob.max(1, keepdim=True)[1].numpy()
        
        if counter.value > currentGap:
            currentGap += checkPointGap
            checkPointCount += 1
            checkPointDict["state_dict"] = model.state_dict()
            saveCheckPoint(checkPointDict, checkPointCount, args)
        
        if args.getValue("render_env"):
            env.render()

        state, reward, done, _ = env.step(action.numpy()[0, 0])
        done = done or episode_length >= args.getValue("max_episode_length")
        reward_sum += reward

        if done:
            #print("Time {}, num steps {}, FPS {:.0f}, episode reward {}, episode length {}".format(time.strftime("%Hh %Mm %Ss",time.gmtime(time.time() - start_time)),counter.value, counter.value / (time.time() - start_time), reward_sum, episode_length))
            writer.add_scalar("performance_curves/reward", reward_sum, counter.value)
            writer.add_scalar("performance_curves/episode_length", episode_length, counter.value)
            writer.add_scalar("policy_characterstics/entropy", episode_policy_entropy/episode_length, counter.value)
            writer.file_writer.flush()
            reward_sum = 0
            episode_policy_entropy = 0.0
            episode_length = 0
            state = env.reset()

        state = torch.from_numpy(state)
