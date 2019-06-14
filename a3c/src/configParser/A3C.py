from a3c.src.algoConfigs.A3C import A3C_DEFAULT_DICT
from copy import deepcopy
import yaml
import os

'''

Parser accepts a valid user Configuration and checks for any error.
If Configuration provided is valid, it replaces the value of keys define in the configuration,
with the default one.

'''

class ConfigParser:

    def __init__(self, config):
        self.configFile = config

        # Check File Exists
        self.isExist(config)

        # Read Configuration File
        self.readConfig(self.configFile)

        # Copy the values of arguments and merge it with a3c Arguments
        self.a3cConfig = deepcopy(A3C_DEFAULT_DICT)
        self.mergeConfig()
    
    def isExist(self, configFile):
        if(os.path.isfile(configFile)):
            return 1
        else:
            raise ValueError("File \"%s\" not found !"%(configFile))

    def readConfig(self, configFile):
        try:
            with open(configFile, "rb") as handle:
                self.userConfig = yaml.load(handle, Loader=yaml.FullLoader)["A3C_Config"]
        except:
            raise ValueError("Invalid configuration file")

    def mergeConfig(self):
        for key in self.userConfig:
            if key not in A3C_DEFAULT_DICT.keys():
                raise ValueError("Invalid Configuration key %s "%(key))
            self.a3cConfig[key] = self.userConfig[key]
        
        return True
        
    def printConfig(self):
        print("=====================================")
        print("A3C Experiment Configuration")
        print("=====================================")
        
        for i, key in enumerate(self.a3cConfig.keys()):
            print("%d. %s : %s"%(i+1, key, self.a3cConfig[key]))
        
        print("=====================================")
    
    def getValue(self, key):

        if key not in self.a3cConfig:
            raise ValueError("Invalid key %s"%(key))
        else:
            return self.a3cConfig[key]