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
from config import parse_arguments
from run_single_inference import SingleImageInference


class RewardEvalCallback(BaseCallback):
    def __init__(self, current_env, eval_env, RL_n_steps, eval_freq, save_path, verbose=0):
        super(RewardEvalCallback, self).__init__(verbose)
        self.current_env = current_env
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.RL_n_steps = RL_n_steps
        self.save_path = save_path

        self.episode_rewards = []
        self.episode_steps = []
        self.current_rewards = 0
        self.current_step = 0
        self.eval_mean_rewards = []
        self.eval_std_rewards = []
        self.eval_mean_episode_length = []
        self.eval_std_episode_length = []
        self.eval_count = 0
        self.best_mean_reward = float('-inf')
        self.best_std_reward = 0
        self.best_eval_count = 0

    def _on_step(self) -> bool:
        self.current_rewards += self.locals['rewards'][0]
        self.current_step += 1

        # Check if the model should be evaluated
        if self.current_env.total_steps % self.eval_freq == 1:
            self.update_eval()

        # Check if the episode has ended
        if self.current_env.done_episode:
            self.episode_rewards.append(self.current_rewards)
            self.episode_steps.append(self.current_step)
            self.current_rewards = 0
            self.current_step = 0
        return True

    def update_eval(self):
        self.eval_count += 1
        mean_reward, std_reward, mean_episode_length, std_episode_length = eval_RL(eval_env=self.eval_env,
                                                                                   model=self.model)
        print(f"Evaluation {self.eval_count}: \nMean Reward = {mean_reward}; STD = {std_reward}")
        self.eval_mean_rewards.append(mean_reward)
        self.eval_std_rewards.append(std_reward)
        self.eval_mean_episode_length.append(mean_episode_length)
        self.eval_std_episode_length.append(std_episode_length)

        # Save the best model if found
        if mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward
            self.best_std_reward = std_reward
            self.best_eval_count = self.eval_count
            print("Current Best Reward!")
            self.model.save(self.save_path + "best_model")


def eval_RL(eval_env, model, eval_episodes=8, deterministic=False):
    eval_env.in_eval = True
    all_episode_rewards = []
    all_episode_length = []
    for episode in range(eval_episodes):
        obs = eval_env.reset()[0]
        episode_reward = 0
        episode_steps_survived = 0
        while True:
            act = model.predict(obs, deterministic=deterministic)[0]
            obs, reward, terminated, truncated, info = eval_env.step(act)
            episode_reward += reward
            episode_steps_survived += 1
            if terminated or truncated:
                break
        all_episode_rewards.append(episode_reward)
        all_episode_length.append(episode_steps_survived)
    mean_reward = np.mean(all_episode_rewards)
    std_reward = np.std(all_episode_rewards)
    mean_episode_length = np.mean(all_episode_length)
    std_episode_length = np.std(all_episode_length)
    # print(f"Mean Reward = {mean_reward}; STD = {std_reward}")
    eval_env.in_eval = False
    return mean_reward, std_reward, mean_episode_length, std_episode_length


def train_RL(train_env, eval_env, model_name, save_path, model_type="PPO", learn_timesteps=20480, RL_n_steps=2048):

    # Define RL model
    model = None
    if model_type == "PPO":
        model = PPO('MultiInputPolicy', train_env, verbose=0, n_steps=RL_n_steps)
    else:
        print(f"The model type {model_type} is not defined!")

    reward_eval_callback = RewardEvalCallback(current_env=train_env, eval_env=eval_env, RL_n_steps=RL_n_steps,
                                              eval_freq=RL_n_steps, save_path=save_path)
    model.learn(total_timesteps=learn_timesteps, callback=reward_eval_callback)

    # Evaluated the final model (which is not evaluated during callback)
    reward_eval_callback.update_eval()

    # Save the last model (optional, maybe unnecessary)
    model.save(save_path + "final_model")

    # Plot the learning curve and evaluation curve
    plot_callback(reward_eval_callback)


def plot_callback(reward_eval_callback):
    # Plot the learning curve
    plt.plot(reward_eval_callback.episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(f'{model_type} Learning Curve: Rewards')
    plt.savefig(save_path + "train_rewards_plot.png")
    plt.clf()

    plt.plot(reward_eval_callback.episode_steps)
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.title(f'{model_type} Learning Curve: Episode Length')
    plt.savefig(save_path + "train_length_plot.png")
    plt.clf()

    # Plot the evaluation curve
    mean_reward_np = np.asarray(reward_eval_callback.eval_mean_rewards)
    std_reward_np = np.asarray(reward_eval_callback.eval_std_rewards)
    plt.title(f"{model_type} Evaluation Curve: Rewards")
    plt.fill_between(range(1, len(mean_reward_np) + 1), mean_reward_np - std_reward_np, mean_reward_np +
                     std_reward_np, alpha=0.2)
    plt.plot(range(1, len(mean_reward_np) + 1), mean_reward_np)
    plt.xlabel(f"Evaluation Count")
    plt.ylabel("Mean Reward")
    plt.savefig(save_path + "eval_rewards_plot.png")
    plt.clf()

    mean_episode_np = np.asarray(reward_eval_callback.eval_mean_episode_length)
    std_episode_np = np.asarray(reward_eval_callback.eval_std_episode_length)
    plt.title(f"{model_type} Evaluation Curve: Episode Length")
    plt.fill_between(range(1, len(mean_reward_np) + 1), mean_episode_np - std_episode_np, mean_episode_np +
                     std_episode_np, alpha=0.2)
    plt.plot(range(1, len(mean_episode_np) + 1), mean_episode_np)
    plt.xlabel(f"Evaluation Count")
    plt.ylabel("Mean Steps")
    plt.savefig(save_path + "eval_length_plot.png")
    plt.clf()


if __name__ == "__main__":

    train_env = WeedDetectionEnv(dataset_dir="./image_data/geok_grouped/train")
    eval_env = WeedDetectionEnv(dataset_dir="./image_data/geok_grouped/valid",
                                group_selection_mode="sequential")  # TODO: set path to valid or test dir

    # Get arguments from command line
    args = parse_arguments()
    model_name = args.model_name  # PPO_test0
    model_type = args.model_type  # PPO
    learn_timesteps = args.learn_timesteps  # 2560
    RL_n_steps = args.RL_n_steps  # 256

    save_path = f"./RL_models/{model_name}/"
    os.makedirs(save_path, exist_ok=True)
    snn_interface = SingleImageInference(
        dataset="geok",
        # Tuple of two numbers: (128, 128), (256, 256), or (512, 512)
        image_resolution=(
            512,
            512,
        ),
        # slim or squeeze
        model_architecture="slim",
        model_path="SNN_models/geok_slim_final.pt",
        # Set to a positive integer to select a specific image from the dataset
        fixed_image=-1,
        # Do you want to generate a mask/image overlay
        save_image=False,
        # Was segmentation model trained using transfer learning
        is_trans=False,
        # Was segmentation model trained with find_best_fitting (utilising
        # model that has the highest difference in iou between widths
        is_best_fitting=False,)

    train_env.set_snn_interface(snn_interface)
    eval_env.set_snn_interface(snn_interface)

    train_RL(train_env, eval_env, model_name, save_path, model_type=model_type, learn_timesteps=learn_timesteps,
             RL_n_steps=RL_n_steps)




