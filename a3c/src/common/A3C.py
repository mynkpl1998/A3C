from a3c.src.policies.A3C import MLPPolicy

import gym
import torch
import torch.nn.functional as F

def single_env_train(args, rank, obs_size, num_actions, info, shared_model):
    
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

            '''
            if done:
                info["episodes"] += 1
            '''

            if done:
                episode_length, epr, eploss = 0, 0, 0
                state = torch.tensor(env.reset())
                
    



