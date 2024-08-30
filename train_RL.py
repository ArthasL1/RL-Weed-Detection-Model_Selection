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
import csv
import random
from image_processing.evaluate_predict_only_avg_iou_model_CNN import CNNModel
import json


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
        self.mean_reward_a0 = 0
        self.mean_reward_a1 = 0
        self.mean_reward_a2 = 0
        self.mean_reward_a3 = 0
        self.mean_reward_optimal = 0
        self.optimal_reward_indices = 0

        self.random_action_mean_rewards = []
        self.random_action_std_rewards = []
        self.random_action_mean_episode_length = []
        self.random_action_std_episode_length = []

        self.eval_baseline_fixed_action()
        self.reward_random_action = 0
        for i in range(5):
            self.reward_random_action += self.eval_baseline_random_action()
        self.mean_reward_random_action = self.reward_random_action / 5

    def _on_step(self) -> bool:
        self.current_rewards += self.locals['rewards'][0]
        self.current_step += 1

        # Check if the model should be evaluated
        if self.current_env.total_steps % self.eval_freq == 1:
            self.update_eval()
            #self.eval_baseline_random_action()

        # Check if the episode has ended
        if self.current_env.done_episode:
            self.episode_rewards.append(self.current_rewards)
            self.episode_steps.append(self.current_step)
            self.current_rewards = 0
            self.current_step = 0
        return True

    def update_eval(self):

        self.eval_env.eval_action_list.append([f"Evaluation {self.eval_count}", ])
        self.eval_env.eval_reward_list.append([f"Evaluation {self.eval_count}", ])
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
            #self.model.save(self.save_path + "best_model")

        self.eval_count += 1
        plot_callback(self)
        plot_baseline(self)
        save_callback_csv(self)
        save_list_csv(self.eval_env.eval_action_list, self.save_path + "eval_actions")
        save_list_csv(self.eval_env.eval_reward_list, self.save_path + "eval_rewards")

    def eval_baseline_random_action(self):
        mean_reward, std_reward, mean_episode_length, std_episode_length = eval_RL_random_action(eval_env=self.eval_env)
        self.random_action_mean_rewards.append(mean_reward)
        self.random_action_std_rewards.append(std_reward)
        self.random_action_mean_episode_length.append(mean_episode_length)
        self.random_action_std_episode_length.append(std_episode_length)
        return mean_reward

    def eval_baseline_fixed_action(self):
        mean_reward_a0, _, _, _, reward_list_a0 = eval_RL_fixed_action(eval_env=self.eval_env, fixed_action=0)
        mean_reward_a1, _, _, _, reward_list_a1 = eval_RL_fixed_action(eval_env=self.eval_env, fixed_action=1)
        mean_reward_a2, _, _, _, reward_list_a2 = eval_RL_fixed_action(eval_env=self.eval_env, fixed_action=2)
        mean_reward_a3, _, _, _, reward_list_a3 = eval_RL_fixed_action(eval_env=self.eval_env, fixed_action=3)
        _, optimal_reward_indices, mean_reward_optimal = process_optimal_reward(reward_list_a0, reward_list_a1,
                                                                                reward_list_a2, reward_list_a3)
        self.mean_reward_a0 = mean_reward_a0
        self.mean_reward_a1 = mean_reward_a1
        self.mean_reward_a2 = mean_reward_a2
        self.mean_reward_a3 = mean_reward_a3
        self.mean_reward_optimal = mean_reward_optimal
        self.optimal_reward_indices = optimal_reward_indices

        return mean_reward_a0, mean_reward_a1, mean_reward_a2, mean_reward_a3, mean_reward_optimal, \
               optimal_reward_indices

    def get_baseline_rewards(self):
        return self.mean_reward_a0, self.mean_reward_a1, self.mean_reward_a2, self.mean_reward_a3, \
               self.mean_reward_optimal, self.optimal_reward_indices

def eval_RL(eval_env, model, deterministic=False):
    eval_env.in_eval = True
    all_episode_rewards = []
    all_episode_length = []
    all_rewards_list = []
    eval_episodes = args.eval_episodes
    for episode in range(eval_episodes):
        rewards_per_step_list = []
        action_list = [f"Episode {episode}", ]
        reward_list = [f"Episode {episode}", ]
        obs = eval_env.reset()[0]
        episode_reward = 0
        episode_steps_survived = 0
        while True:
            act = model.predict(obs, deterministic=deterministic)[0]
            action_list.append(act.item())
            obs, reward, terminated, truncated, info = eval_env.step(act)
            reward_list.append(round(reward, 2))
            episode_reward += reward
            rewards_per_step_list.append(reward)
            episode_steps_survived += 1
            if terminated or truncated:
                break
        all_episode_rewards.append(episode_reward)
        all_episode_length.append(episode_steps_survived)
        eval_env.eval_action_list.append(action_list)
        eval_env.eval_reward_list.append(reward_list)
        all_rewards_list.append(rewards_per_step_list)
    mean_reward = np.mean(all_episode_rewards)
    std_reward = np.std(all_episode_rewards)
    mean_episode_length = np.mean(all_episode_length)
    std_episode_length = np.std(all_episode_length)
    # print(f"Mean Reward = {mean_reward}; STD = {std_reward}")
    eval_env.in_eval = False
    return mean_reward, std_reward, mean_episode_length, std_episode_length

def eval_RL_fixed_action(eval_env, fixed_action):
    eval_env.in_eval = True
    all_episode_rewards = []
    all_episode_length = []
    all_rewards_list = []
    eval_episodes = args.eval_episodes
    for episode in range(eval_episodes):
        rewards_per_step_list = []
        obs = eval_env.reset()[0]
        episode_reward = 0
        episode_steps_survived = 0
        while True:
            obs, reward, terminated, truncated, info = eval_env.step(fixed_action)
            episode_reward += reward
            rewards_per_step_list.append(reward)
            episode_steps_survived += 1
            if terminated or truncated:
                break
        all_episode_rewards.append(episode_reward)
        all_episode_length.append(episode_steps_survived)
        all_rewards_list.append(rewards_per_step_list)
    mean_reward = np.mean(all_episode_rewards)
    std_reward = np.std(all_episode_rewards)
    mean_episode_length = np.mean(all_episode_length)
    std_episode_length = np.std(all_episode_length)
    # print(f"Mean Reward = {mean_reward}; STD = {std_reward}")
    eval_env.in_eval = False
    return mean_reward, std_reward, mean_episode_length, std_episode_length, all_rewards_list

def process_optimal_reward(L0, L1, L2, L3):
    # Convert the lists to numpy arrays for easier manipulation
    L0 = np.array(L0)
    L1 = np.array(L1)
    L2 = np.array(L2)
    L3 = np.array(L3)

    # Stack the lists along a new dimension to facilitate comparison
    stacked = np.stack([L0, L1, L2, L3], axis=0)

    # Find the maximum values across the new dimension (axis=0)
    max_values = np.max(stacked, axis=0)

    # Find the indices of the maximum values across the new dimension (axis=0)
    max_indices = np.argmax(stacked, axis=0)

    # Calculate the sum of each sublist (row-wise sum) to reduce the 2D list to 1D
    row_sums = np.sum(max_values, axis=1)

    # Calculate the average of the summed values
    average_value = np.mean(row_sums)

    return max_values.tolist(), max_indices.tolist(), average_value

def eval_RL_random_action(eval_env, action_space=[0, 1, 2, 3]):
    eval_env.in_eval = True
    all_episode_rewards = []
    all_episode_length = []
    eval_episodes = args.eval_episodes
    for episode in range(eval_episodes):
        obs = eval_env.reset()[0]
        episode_reward = 0
        episode_steps_survived = 0
        while True:
            act = random.choice(action_space)
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

def save_list_csv(list_to_save, file_name):
    with open(f'{file_name}.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for item in list_to_save:
            writer.writerow([item])


def train_RL(train_env, eval_env, model_name, save_path, eval_freq=2048, RL_model_type="PPO", learn_timesteps=20480,
             RL_n_steps=2048, minibatch_size=64):

    # Define RL model
    model = None
    if RL_model_type == "PPO":
        model = PPO('MlpPolicy',  # MlpPolicy, CnnPolicy, MultiInputPolicy
                    train_env,
                    verbose=0,
                    n_steps=RL_n_steps,
                    batch_size=minibatch_size,
                    ent_coef=args.ent_coef,
                    gamma=args.gamma,
                    learning_rate=args.RL_learning_rate,
                    )
        print(model.policy)
    else:
        print(f"The model type {RL_model_type} is not defined!")

    reward_eval_callback = RewardEvalCallback(current_env=train_env, eval_env=eval_env, RL_n_steps=RL_n_steps,
                                              eval_freq=eval_freq, save_path=save_path)
    model.learn(total_timesteps=learn_timesteps, callback=reward_eval_callback)

    # Evaluated the final model (which is not evaluated during callback)
    reward_eval_callback.update_eval()
    #reward_eval_callback.eval_baseline_random_action()

    # Save the last model (optional, maybe unnecessary)
    #model.save(save_path + "final_model")

    # Plot and save the learning curve and evaluation curve
    plot_callback(reward_eval_callback)
    plot_baseline(reward_eval_callback)
    save_callback_csv(reward_eval_callback)
    save_list_csv(eval_env.eval_action_list, save_path + "eval_actions")
    save_list_csv(eval_env.eval_reward_list, save_path + "eval_rewards")

def save_callback_csv(reward_eval_callback):
    save_list_csv(reward_eval_callback.episode_rewards, save_path + "train_rewards")
    save_list_csv(reward_eval_callback.episode_steps, save_path + "train_steps")
    save_list_csv(reward_eval_callback.eval_mean_rewards, save_path + "eval_mean_rewards")
    save_list_csv(reward_eval_callback.eval_std_rewards, save_path + "eval_std_rewards")
    save_list_csv(reward_eval_callback.eval_mean_episode_length, save_path + "eval_mean_steps")
    save_list_csv(reward_eval_callback.eval_std_episode_length, save_path + "eval_std_steps")

def plot_baseline(reward_eval_callback):
    random_action_mean_rewards_np = np.asarray(reward_eval_callback.random_action_mean_rewards)
    mean_reward_np = np.asarray(reward_eval_callback.eval_mean_rewards)
    std_reward_np = np.asarray(reward_eval_callback.eval_std_rewards)
    mean_reward_a0, mean_reward_a1, mean_reward_a2, mean_reward_a3, mean_reward_optimal, optimal_reward_indices = \
        reward_eval_callback.get_baseline_rewards()
    # reward_random_action = 0
    # for i in range(3):
    #     reward_random_action += reward_eval_callback.eval_baseline_random_action()
    # mean_reward_random_action = reward_random_action / 3
    plt.plot(range(len(mean_reward_np)), mean_reward_np, label='RL Model Learning Curve')
    plt.fill_between(range(len(mean_reward_np)), mean_reward_np - std_reward_np, mean_reward_np +
                     std_reward_np, alpha=0.2)
    # plt.plot(range(len(mean_reward_np)),random_action_mean_rewards_np,
    #          label='Random Action Baseline')
    plt.axhline(y=reward_eval_callback.mean_reward_random_action, color='orange', linestyle='--',
                label='Random Action')
    plt.axhline(y=mean_reward_a0, color='red', linestyle='--',
                label='Fixed Action 0')
    plt.axhline(y=mean_reward_a1, color='green', linestyle='--',
                label='Fixed Action 1')
    plt.axhline(y=mean_reward_a2, color='purple', linestyle='--',
                label='Fixed Action 2')
    plt.axhline(y=mean_reward_a3, color='brown', linestyle='--',
                label='Fixed Action 3')
    plt.axhline(y=mean_reward_optimal, color='black', linestyle='--',
                label='Optimal Action')
    plt.title('RL Model Learning Curve vs. Baselines')
    plt.xlabel('Evaluation Count')
    plt.ylabel('Mean Reward')
    plt.legend(fontsize='small')
    plt.savefig(save_path + "eval_rewards_plot_vs_baselines.png")
    plt.clf()

    save_list_csv(optimal_reward_indices, save_path + "optimal_action_choice")


def plot_callback(reward_eval_callback):
    # Plot the learning curve
    plt.plot(reward_eval_callback.episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(f'{RL_model_type} Learning Curve: Rewards')
    plt.savefig(save_path + "train_rewards_plot.png")
    plt.clf()

    plt.plot(reward_eval_callback.episode_steps)
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.title(f'{RL_model_type} Learning Curve: Episode Length')
    plt.savefig(save_path + "train_length_plot.png")
    plt.clf()

    # Plot the evaluation curve
    mean_reward_np = np.asarray(reward_eval_callback.eval_mean_rewards)
    std_reward_np = np.asarray(reward_eval_callback.eval_std_rewards)
    plt.title(f"{RL_model_type} Evaluation Curve: Rewards")
    plt.fill_between(range(len(mean_reward_np)), mean_reward_np - std_reward_np, mean_reward_np +
                     std_reward_np, alpha=0.2)
    plt.plot(range(len(mean_reward_np)), mean_reward_np)
    plt.xlabel(f"Evaluation Count")
    plt.ylabel("Mean Reward")
    plt.savefig(save_path + "eval_rewards_plot.png")
    plt.clf()

    mean_episode_np = np.asarray(reward_eval_callback.eval_mean_episode_length)
    std_episode_np = np.asarray(reward_eval_callback.eval_std_episode_length)
    plt.title(f"{RL_model_type} Evaluation Curve: Episode Length")
    plt.fill_between(range(len(mean_reward_np)), mean_episode_np - std_episode_np, mean_episode_np +
                     std_episode_np, alpha=0.2)
    plt.plot(range(len(mean_episode_np)), mean_episode_np)
    plt.xlabel(f"Evaluation Count")
    plt.ylabel("Mean Steps")
    plt.savefig(save_path + "eval_length_plot.png")
    plt.clf()


def save_config(args):
    config_filename = os.path.join(save_path, f"config.json")
    with open(config_filename, 'w') as f:
        json.dump(vars(args), f, indent=4)



if __name__ == "__main__":
    # Get arguments from command line
    args = parse_arguments()
    model_name = args.model_name  # PPO_test0
    RL_model_type = args.RL_model_type  # PPO
    learn_timesteps = args.learn_timesteps  # 20480
    RL_n_steps = args.RL_n_steps  # 2048
    snn_model_type = args.snn_model_type  # slim, squeeze
    output_layer = args.output_layer  # None, conv2, avg_iou
    eval_freq = args.eval_freq  # 2048
    minibatch_size = args.minibatch_size  # 64

    snn_interface = SingleImageInference(
        dataset=args.snn_dataset,  # geok, tobacco
        # Tuple of two numbers: (128, 128), (256, 256), or (512, 512)
        image_resolution=(
            512,
            512,
        ),
        # slim or squeeze
        model_architecture=snn_model_type,
        model_path=f"SNN_models/{args.snn_dataset}_{snn_model_type}_final.pt",
        # Set to a positive integer to select a specific image from the dataset
        fixed_image=-1,
        # Do you want to generate a mask/image overlay
        save_image=False,
        # Was segmentation model trained using transfer learning
        is_trans=False,
        # Was segmentation model trained with find_best_fitting (utilising
        # model that has the highest difference in iou between widths
        is_best_fitting=False,)

    #snn_interface = None

    iou_predict_model = CNNModel()
    iou_predict_model.load_state_dict(torch.load('./image_processing/best_predict_only_avg_iou_model_CNN.pth'))
    iou_predict_model.eval()

    train_env = WeedDetectionEnv(dataset_dir="./image_data/geok_grouped_cleaned_iou0_no_rename/valid",
                                 iou_result_path=f"./image_data/geok_grouped_cleaned_iou0_no_rename/{snn_model_type}_iou_results.json",
                                 snn_dataset=args.snn_dataset,
                                 snn_interface=snn_interface,
                                 snn_model_type=snn_model_type,
                                 output_layer=output_layer,  # conv2, avg_iou, None
                                 group_selection_mode="random",
                                 iou_predict_model=iou_predict_model
                                 )
    eval_env = WeedDetectionEnv(dataset_dir="./image_data/geok_grouped_cleaned_iou0_no_rename/valid",
                                iou_result_path=f"./image_data/geok_grouped_cleaned_iou0_no_rename/{snn_model_type}_iou_results.json",
                                snn_dataset=args.snn_dataset,
                                snn_interface=snn_interface,
                                snn_model_type=snn_model_type,
                                output_layer=output_layer,
                                group_selection_mode="sequential",
                                iou_predict_model=iou_predict_model
                                )



    save_path = f"./RL_models/{model_name}/"
    os.makedirs(save_path, exist_ok=True)

    save_config(args)

    train_RL(train_env, eval_env, model_name, save_path,
             eval_freq=eval_freq,
             RL_model_type=RL_model_type,
             learn_timesteps=learn_timesteps,
             RL_n_steps=RL_n_steps,
             minibatch_size=minibatch_size)




