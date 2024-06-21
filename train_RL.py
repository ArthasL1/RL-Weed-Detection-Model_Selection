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
    def __init__(self, verbose=0):
        super(RewardCallback, self).__init__(verbose)
        self.episode_rewards = []

    def _on_step(self) -> bool:
        # Check if the episode has ended
        if self.locals['dones']:
            print(self.locals['done'], self.locals['dones'], self.locals['rewards'])
            # Append the episode reward to the list
            episode_reward = np.sum(self.locals['rewards'])
            self.episode_rewards.append(episode_reward)
        return True


if __name__ == "__main__":
    env = WeedDetectionEnv()
    model = PPO('MultiInputPolicy', env, verbose=0)
    reward_callback = RewardCallback()
    model.learn(total_timesteps=100, callback=reward_callback)
    print(env.total_episodes, env.total_steps, env.episode_in_process)
    # model.save("model0")
    #
    # # Plot the learning curve
    # plt.plot(reward_callback.episode_rewards)
    # plt.xlabel('Episode')
    # plt.ylabel('Reward')
    # plt.title('Learning Curve')
    # plt.show()


