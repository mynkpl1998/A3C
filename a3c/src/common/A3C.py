from a3c.src.policies.A3C import MLPPolicy

import gym
import torch
import numpy as np
import torch.nn.functional as F
from scipy.signal import lfilter

discount = lambda x, gamma : lfilter([1], [1, -gamma], x[::-1])[::-1]

def cost_func(args, values, logps, actions, rewards):
    np_values = values.view(-1).data.numpy()
    
    '''
    Gradient Calculation
    Web Link : https://lilianweng.github.io/lil-log/assets/images/general_form_policy_gradient.png
    
    '''
    delta_t = rewards + (args.getValue("gamma") * np_values[1:]) - np_values[:-1]
    logpys = logps.gather(1, torch.tensor(actions).view(-1,1))
    
    '''
    GAE - Generalized Advantage Estimation
    Web Link : https://www.google.com/url?sa=i&rct=j&q=&esrc=s&source=images&cd=&ved=2ahUKEwjBy5WploDjAhWSTX0KHbYYApcQjRx6BAgBEAU&url=https%3A%2F%2Fstats.stackexchange.com%2Fquestions%2F367339%2Fgeneralized-advantage-estimation&psig=AOvVaw3D9TVEZgYJfrwpPrawyonB&ust=1561398900407464

    gae = \sum_{l=0}^{\infty} (\gamma * \lambda) ^ {l} * delta_{t+1}
    '''
    gen_adv_estimates = discount(delta_t, args.getValue("gamma") * args.getValue("gae-lambda"))
    policy_loss = -(logpys.view(-1) * torch.FloatTensor(gen_adv_estimates.copy())).sum()
    
    # L2 Loss of value estimator
    rewards[-1] += args.getValue("gamma") * np_values[-1]
    discounted_r = discount(np.asarray(rewards), args.getValue("gamma"))
    discounted_r = torch.tensor(discounted_r.copy(), dtype=torch.float32)
    value_loss = 0.5 * (discounted_r - values[:-1,0]).pow(2).sum()

    # Policy Entropy loss
    entropy_loss = (-logps * torch.exp(logps)).sum()
    return policy_loss + (args.getValue("value-loss-coef") * value_loss) - (args.getValue('entropy-coef') * entropy_loss)


def single_env_train(args, rank, obs_size, num_actions, info, shared_model, shared_optimizer):
    
    env = None

    if(args.getValue("env_type") == "gym"):
        env = gym.make(args.getValue("env_name"))
    else:
        raise ValueError("Only gym environments are supported for now...")

    # Seed the envs and PyTorch
    env.seed(args.getValue("seed_offset") + rank)
    torch.manual_seed(args.getValue("pySeed") + rank)

    # Local Model
    model = MLPPolicy(obs_size, num_actions, args.getValue("memsize"), args.getValue("policy_hiddens"))
    
    state = torch.tensor(env.reset())
    episode_length, epr, eploss, done = 0, 0, 0, True

    
    while info["frames"] <= args.getValue("num_training_frames"):
        
        '''
        At the start of training each env updates its local copy of weights from global copy    
        '''
        
        model.load_state_dict(shared_model.state_dict())
        hx = torch.zeros(1, args.getValue("memsize")) if done else hx.detach()
        
        values, logps, actions, rewards = [], [], [], []

        for step in range(args.getValue("num_steps")):
            episode_length += 1
            value, logit, hx = model.forward(state.view(1, obs_size).float(), hx)
            logp = F.log_softmax(logit, dim=-1)

            action = torch.exp(logp).multinomial(num_samples=1).data[0]
            state, reward, done, _ = env.step(action.numpy()[0])

            if (args.getValue("render_env")):
                env.render()
            
            state = torch.tensor(state)
            epr += reward
            done = done or episode_length >= args.getValue("max_episode_length")

            if done:
                info["episodes"] += 1
                coef = 1 if info["episodes"][0] == 1 else (1 - args.getValue("moving_avg_coef"))
                info["run_epr"].mul_(1 - coef).add_(coef * epr)
                info["run_loss"].mul_(1 - coef).add_(coef * eploss)


            if done:
                episode_length, epr, eploss = 0, 0, 0
                state = torch.tensor(env.reset())
            
            values.append(value)
            logps.append(logp)
            actions.append(action)
            rewards.append(reward)

        #next_value = torch.zeros(1,1)
        next_value = torch.zeros(1, 1) if done else model.forward(state.view(1, obs_size).float(), hx)[0]
        values.append(next_value.detach()) # This is a target hence we need to detach

        '''
        
        torch.cat merges the list of torch tensor to single tensor
        Example :
        a = [torch.tensor([1]), torch.tensor([2])]
        >>> torch.cat(a)
        tensor([1, 2])
        
        '''
        print("Reward Exp. Avg : %.2f, Loss Exp. Avg : %.2f"%(info["run_epr"].item(), info["run_loss"].item()))
        loss = cost_func(args, torch.cat(values), torch.cat(logps), torch.cat(actions), np.asarray(rewards))
        eploss += loss.item()
        shared_optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.getValue("grad-norm"))
        
        for param, shared_param in zip(model.parameters(), shared_model.parameters()):
            if shared_param.grad is None: shared_param._grad = param.grad
        
        shared_optimizer.step()
        