from a3c.src.common.rollouts import rollouts

if __name__ == "__main__":

    path = "/home/mayank/cartpole/saved_models/checkpoint-1"
    testEnv = rollouts(path)
