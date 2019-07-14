import torch
from a3c.src.common.utils import checkFileExists

class rollouts:

    def __init__(self, checkpointPath):
        self.checkpointDict = self.loadCheckpoint(checkpointPath)
        self.model = self.reBuildModel(self.checkpointDict)
        self.args = self.checkpointDict["args"]
        self.episode_data = None
        self.episode_step = 0
        self.prevState = None
        self.nextState = None

    def reBuildModel(self, checkpointDict):
        model = checkpointDict["model"]
        model.load_state_dict(checkpointDict["state_dict"])
        
        for parameter in model.parameters():
            parameter.requires_grad = False
        
        model = model.eval()
        return model

    def loadCheckpoint(self, checkpointPath):
        if checkFileExists(checkpointPath):
            return torch.load(checkpointPath)
        else:
            raise ValueError("Checkpoint \"" + checkpointPath + "\" not found !")

    def reset(self):
        self.episode_data = {}
        self.steps = 0


