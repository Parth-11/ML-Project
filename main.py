import argparse
from snakeql import SnakeQAgent
from visualsnake import VisualSnake
import time

parser = argparse.ArgumentParser()
parser.add_argument('--episode', type=str, help="The episode number to display", required=False)

args = parser.parse_args()

if args.episode == None:
    agent = SnakeQAgent()

    agent.train()
    filenames = [f"graphs/episode_scores{int(time.time())}.png", 
                 f"graphs/survival_time{int(time.time())}.png", 
                 f"graphs/epsilon_decay{int(time.time())}.png", 
                 f"graphs/q_value_convergence{int(time.time())}.png"]
    agent.plot(filenames)
    # agent.get_size()

else:
    agent = VisualSnake()

    agent.run_game(args.episode)