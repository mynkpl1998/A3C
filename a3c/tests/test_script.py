import pytest
import yaml
from a3c.src.configParser.configParser import ConfigParser

class TestconfigParser():

    def runAll(self):
        self.valid_config_t1()
        self.valid_config_t2()
        self.merge_config_t1()
        self.merge_config_t2()
    
    def valid_config_t1(self):
        # Test - 1
        self.config_one = {"test1" : {'env_type': 'gym'}}
        with open("test.yml", "w") as handle:
            yaml.dump(self.config_one, handle)

        with pytest.raises(ValueError) as error:
            ConfigParser("test.yml")

        assert str(error.value) == "Invalid configuration file"
    

    def valid_config_t2(self):
        # Test - 2
        self.config_two = {"A3C_Config": {"env_type": "gym"}}
        with open("test.yml", "w") as handle:
            yaml.dump(self.config_two, handle)
        
        out = ConfigParser("test.yml")
        assert out.userConfig == self.config_two["A3C_Config"]

    def merge_config_t1(self): 
        config = {"A3C_Config" : {"env_type": "gym", "env_processes": 100}}
        
        with open("test.yml", "w") as handle:
            yaml.dump(config, handle)
        
        assert ConfigParser("test.yml").mergeConfig() == True
        
    def merge_config_t2(self): 
        config = {"A3C_Config" : {"env_type": "gym", "env_processes": 100, "abc": "Henlo"}}
        
        with open("test.yml", "w") as handle:
            yaml.dump(config, handle)
    
        with pytest.raises(ValueError) as error:
            ConfigParser("test.yml").mergeConfig()        
        #print(str(error.value) == "Invalid Configuration key abc ")
        assert str(error.value) == "Invalid Configuration key abc "

def test_runner():
    TestconfigParser().runAll()
    