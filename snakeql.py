import numpy as np
import random
from snake_no_visual import LearnSnake
import pickle
import matplotlib.pyplot as plt

class SnakeQAgent():
    def __init__(self):
        self.discount_rate = 0.98
        self.learning_rate = 0.1
        self.eps = 0.7
        self.eps_discount = 0.998
        self.min_eps = 0.001
        self.num_episodes = 100
        self.start_table = 'pickle/q_table.pickle'
        if self.start_table == None:
            self.table = np.zeros((2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4))
        else:
            with open(self.start_table,'rb') as f:
                self.table = pickle.load(f)
        self.env = LearnSnake()
        self.score = []
        self.survived = []
        self.episode_scores = []
        self.episode_survivved = []
        self.epsilon_values = []
        self.average_q_values = []

    def get_size(self):
        print(self.table)
        print(self.table.shape)
        print(self.table.size)
    # epsilon-greedy action choice
    def get_action(self, state):
        if random.random() < self.eps:
            return random.choice([0, 1, 2, 3])
        
        return np.argmax(self.table[state])
    
    def train(self):
        for i in range(1, self.num_episodes + 1):
            self.env  = LearnSnake()
            steps_without_food = 0
            length = self.env.snake_length
            
            episode_score = 0
            episode_survived = 0

            # print updates
            if i % 25 == 0:
                print(f"Episodes: {i}, score: {np.mean(self.score)}, survived: {np.mean(self.survived)}, eps: {self.eps}, lr: {self.learning_rate}")
                self.score = []
                self.survived = []
               
            # occasionally save latest model
            if i%1000==0:
                with open(f'pickle/{i}.pickle', 'wb') as file:
                    pickle.dump(self.table, file)
                # self.plot()
                # plt.savefig(f"graphs/episode_{i}.png")

                
            current_state = self.env.get_state()
            self.eps = max(self.eps * self.eps_discount, self.min_eps)
            done = False
            while not done:
                action = self.get_action(current_state)
                new_state, reward, done = self.env.step(action)
                
                # Bellman Equation Update
                self.table[current_state][action] = (1 - self.learning_rate)\
                    * self.table[current_state][action] + self.learning_rate\
                    * (reward + self.discount_rate * max(self.table[new_state])) 
                current_state = new_state
                
                episode_score = self.env.snake_length - 1
                episode_survived += 1
                steps_without_food += 1
                
                if length != self.env.snake_length:
                    length = self.env.snake_length
                    steps_without_food = 0
                if steps_without_food == 1000:
                    break
            
            self.episode_scores.append(episode_score)
            self.episode_survivved.append(episode_survived)
            self.epsilon_values.append(self.eps)
            avg = np.mean(self.table)
            self.average_q_values.append(avg)
            self.score.append(self.env.snake_length - 1)
            self.survived.append(self.env.survived)

    def plot(self, filenames, downsample_factor=100, moving_avg_window=50):
        if len(filenames) != 4:
            raise ValueError("Please provide exactly 4 filenames for saving the plots.")

        def moving_average(data, window_size):
            return np.convolve(data, np.ones(window_size) / window_size, mode="valid")

        def downsample(data, factor):
            return data[::factor]

        # Plot 1: Episode Scores (with smoothing)
        plt.figure(figsize=(6, 4))
        scores_smoothed = moving_average(self.episode_scores, moving_avg_window)
        plt.plot(scores_smoothed, label="Score (Smoothed)", alpha=0.8)
        plt.xlabel("Episodes")
        plt.ylabel("Score")
        plt.title("Episode Score (Smoothed)")
        plt.legend()
        plt.grid(True)
        plt.ylim([0, max(scores_smoothed) * 1.2])
        plt.savefig(filenames[0])
        plt.close()

        # Plot 2: Survival Time
        plt.figure(figsize=(6, 4))
        survival_downsampled = downsample(self.episode_survivved, downsample_factor)
        plt.plot(survival_downsampled, label="Survival Time", color="orange", alpha=0.8)
        plt.xlabel("Episodes")
        plt.ylabel("Time")
        plt.title("Survival Time (Downsampled)")
        plt.legend()
        plt.grid(True)
        plt.ylim([0, max(self.episode_survivved) * 1.2])
        plt.savefig(filenames[1])
        plt.close()

        # Plot 3: Epsilon Decay
        plt.figure(figsize=(6, 4))
        epsilon_downsampled = downsample(self.epsilon_values, downsample_factor)
        plt.plot(epsilon_downsampled, label="Epsilon Decay", color="green", alpha=0.8)
        plt.xlabel("Episodes")
        plt.ylabel("Epsilon")
        plt.title("Epsilon Decay (Downsampled)")
        plt.legend()
        plt.grid(True)
        plt.ylim([0, 1])
        plt.savefig(filenames[2])
        plt.close()

        # Plot 4: Q-value Convergence (with smoothing)
        plt.figure(figsize=(6, 4))
        q_values_smoothed = moving_average(self.average_q_values, moving_avg_window)
        plt.plot(q_values_smoothed, label="Average Q-value (Smoothed)", color="red", alpha=0.8)
        plt.xlabel("Episodes")
        plt.ylabel("Average Q-value")
        plt.title("Q-value Convergence (Smoothed)")
        plt.legend()
        plt.grid(True)
        plt.ylim([min(q_values_smoothed) * 0.9, max(q_values_smoothed) * 1.2])
        plt.savefig(filenames[3])
        plt.close()


        
