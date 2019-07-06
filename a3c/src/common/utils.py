import os
import shutil
import time

def logEssentials(dir_path, exp_name):

    # Create directory first
    if not os.path.exists(dir_path+exp_name):
        os.mkdir(dir_path+exp_name)
    else:
        print("Do you want to remove existing experiment logs and data ? [Y/y]")
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
    os.system(launchCmd)
    print("tensorboard Started at localhost:6006")
