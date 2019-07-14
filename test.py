import argparse
from a3c.src.common.rollouts import rollouts

parser = argparse.ArgumentParser(description="Script to run trained agent")
parser.add_argument("--checkpoint-file", type=str, help="checkpoint file to load")
parser.add_argument("--num-episodes", type=int, default=10, help="number of episodes to run, (default = 10)")
parser.add_argument("--render", type=int, default=1, help="render the environment (1-true, 0-false), (default = 1)")

if __name__ == "__main__":

    args = parser.parse_args()
    testEnv = rollouts(args.checkpoint_file)

    for i in range(0, args.num_episodes):
        render = args.render == 1

        res = testEnv.runEpisode(render)
        print("Finished Episode : ", i+1, "Reward : ", res["cum_reward"], "Episode Length : ", res["episode_length"])
    
    testEnv.env.close()