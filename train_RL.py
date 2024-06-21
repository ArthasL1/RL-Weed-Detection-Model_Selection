from env import *
import numpy as np
from stable_baselines3 import PPO, SAC, DDPG, A2C, TD3, DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
import torch
from stable_baselines3.common.evaluation import evaluate_policy
import matplotlib.pyplot as plt
import os
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

class RewardCallback(BaseCallback):
    def __init__(self, current_env, RL_n_steps, verbose=0):
        super(RewardCallback, self).__init__(verbose)
        self.current_env = current_env
        self.episode_rewards = []
        self.episode_steps = []
        self.current_rewards = 0
        self.current_step = 0

    def _on_step(self) -> bool:
        self.current_rewards += self.locals['rewards'][0]
        self.current_step += 1
        # Check if the episode has ended
        if self.current_env.done_episode:
            self.episode_rewards.append(self.current_rewards)
            self.episode_steps.append(self.current_step)
            self.current_rewards = 0
            self.current_step = 0
        return True

def train_RL(env, model_name, save_path, model_type="PPO", learn_timesteps=20480, RL_n_steps=2048):

    # Define RL model
    model = None
    if model_type == "PPO":
        model = PPO('MultiInputPolicy', env, verbose=0, n_steps=RL_n_steps)
    else:
        print(f"The model type {model_type} is not defined!")

    reward_callback = RewardCallback(current_env=env, RL_n_steps=RL_n_steps)
    model.learn(total_timesteps=learn_timesteps, callback=reward_callback)

    model.save(save_path + "final_model")
    # Plot the learning curve
    plt.plot(reward_callback.episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Learning Curve')
    plt.savefig(save_path + "rewards_plot.png")
    plt.clf()

    plt.plot(reward_callback.episode_steps)
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.title('Steps Curve')
    plt.savefig(save_path + "steps_plot.png")
    plt.clf()



if __name__ == "__main__":
    env = WeedDetectionEnv()
    model_name = "PPO_test0"
    save_path = f"./RL_models/{model_name}/"
    os.makedirs(save_path, exist_ok=True)
    train_RL(env, model_name, save_path, model_type="PPO", learn_timesteps=20480, RL_n_steps=2048)




