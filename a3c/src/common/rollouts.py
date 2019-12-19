import torch
import torch.nn.functional as F
from a3c.src.common.utils import saliencyMaps
from a3c.src.common.utils import checkFileExists

class rollouts:

    def __init__(self, checkpointPath, maps=0):
        self.checkpointDict = self.loadCheckpoint(checkpointPath)
        self.model = self.reBuildModel(self.checkpointDict)
        self.args = self.checkpointDict["args"]
        self.vecEnv = self.checkpointDict["env"]
        self.env = self.vecEnv.createEnv(self.args.getValue("normalize_state"))
        self.episode_data = None
        self.episode_step = 0
        self.prevState = None
        self.nextState = None
        self.maps = maps
        if self.maps == 1 and self.args.getValue('policy-type') == 'cnn':
            self.mapObject = self.getMapsObject()
    
    def getMapsObject(self,):
        return saliencyMaps(self.model)

    def reBuildModel(self, checkpointDict):
        '''
        Doc String : Rebuilds the existing model from the checkpoint and copies the weights.
        input args : takes checkpoint dict as the input
        return : returns the model from checkpoitn where paramerts are freezed model.
        '''
        
        model = checkpointDict["model"]
        model.load_state_dict(checkpointDict["state_dict"])
        
        for parameter in model.parameters():
            parameter.requires_grad = False
        
        model = model.eval()
        return model

    def loadCheckpoint(self, checkpointPath):
        '''
        Doc String : Loads the saved checkpoint from the provided path. If path is not valid then raises the value error
        Input Args : checkpoint file path
        Returns : if successful returns the checkpoint dictionary else raises error. 
        '''

        if checkFileExists(checkpointPath):
            return torch.load(checkpointPath)
        else:
            raise ValueError("Checkpoint \"" + checkpointPath + "\" not found !")

    def runEpisode(self, render=True):
        
        history = {'ins': [], 'logits': [], 'values': [], 'outs':[], 'hx': []}
        state = self.env.reset()
        if self.args.getValue('policy-type') == 'cnn':
            state = torch.from_numpy(state).transpose(2, 0)
        else:
            state = torch.from_numpy(state)
        episode_data = {}

        reward_sum = 0
        done = True

        episode_length = 0

        while True:

            episode_length += 1
            
            # Logging Step
            stepStr = "step_%d"%(episode_length)
            episode_data[stepStr] = {}

            if done:
                hx = torch.zeros(1, self.args.getValue("memsize"))
            else:
                hx = hx.detach()
            
            episode_data[stepStr]["hx"] = hx.clone().numpy()
            episode_data[stepStr]["state"] = state.clone().numpy()
        
            value, logit, hx = self.model((state.unsqueeze(0).float(), hx))
            if self.args.getValue('policy-type') == 'cnn' and self.maps == 1:
                self.mapObject.forward((state.unsqueeze(0).float(), hx))

            prob = F.softmax(logit, dim=-1)
            action = prob.multinomial(num_samples=1).detach()
        
            if render and self.args.getValue('env_type') == 'gym':
                self.env.render()

            episode_data[stepStr]["act_dist"] = prob.clone().numpy()
            episode_data[stepStr]["action"] = action.numpy()[0,0]

            state, reward, done, _ = self.env.step(action.numpy()[0, 0])
            done = done or episode_length >= self.args.getValue("max_episode_length")
            reward_sum += reward

            if self.args.getValue('policy-type') == 'cnn':
                state = torch.from_numpy(state).transpose(2, 0)
            else:
                state = torch.from_numpy(state)
        
            episode_data[stepStr]["reward"] = reward
            episode_data[stepStr]["next_state"] = state.clone().numpy()
            episode_data[stepStr]["done"] = done

            # Save info
            history['ins'].append(state.data.numpy())
            history['hx'].append(hx.squeeze(0).data.numpy())
            history['logits'].append(logit.data.numpy()[0])
            history['values'].append(value.data.numpy()[0])
            history['outs'].append(prob.data.numpy()[0])
            

            if done:
                episode_data["episode_length"] = episode_length
                episode_data["cum_reward"] = reward_sum
                break
        
        return episode_data, history