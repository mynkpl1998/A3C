import os
import shutil
import time
from huepy import bold, bad, red, blue, info

def warnMessage(msgStr):
    print(bad(blue(red(msgStr))))

def infoMessage(msgStr):
    print(info(bold(blue(msgStr))))

def checkFileExists(path):
    return os.path.exists(path)
    
def logEssentials(dir_path, exp_name):

    # Create directory first
    if not os.path.exists(dir_path+exp_name):
        os.mkdir(dir_path+exp_name)
    else:
        warnMessage("Do you want to remove existing experiment logs and data ? [Y/y]")
        res = input()
        if res == 'Y' or res == 'y':
            shutil.rmtree(dir_path+exp_name)
            os.mkdir(dir_path+exp_name)
        else:
            return True
    
    os.mkdir(dir_path+exp_name+"/logs")
    os.mkdir(dir_path+exp_name+"/saved_models")
    return False

def launchTensorboard(launch_dir):
    launchCmd = "tensorboard --logdir "+launch_dir
    infoMessage("Starting tesorboard at localhost:6006")
    os.system(launchCmd)
    
def loadCustomEnv(modulePath):
    print(modulePath)