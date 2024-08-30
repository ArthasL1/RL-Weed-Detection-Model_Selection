import numpy as np
import pdb
import gymnasium as gym
import random
import torch
from gymnasium import spaces
import os
import cv2
from image_processing.evaluate_predict_only_avg_iou_model_CNN import predict_avg_iou
import json
from stable_baselines3.common.preprocessing import is_image_space


# SEED = 2024
SEED = None

if SEED is not None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

class WeedDetectionEnv(gym.Env):
    def __init__(self,
                 dataset_dir="./image_data/geok_grouped_cleaned_iou0_no_rename/train",
                 iou_result_path="./image_data/geok_grouped_cleaned_iou0_no_rename/iou_results.json",
                 # None or "./image_data/geok_grouped_cleaned_iou0_no_rename/iou_results.json"
                 snn_dataset="geok",  # The name of the SNN dataset, such as "geok", "tobacco"
                 ini_energy=1800,
                 images_per_group=10,
                 image_resolution=512,
                 group_selection_mode="random",  # ["random", "sequential"]
                 snn_model_type="slim",  # ["slim", "squeeze"]
                 noise_type="gaussian",  # Noise of energy consumption. ["gaussian", "uniform"]
                 noise_level=0.05,  # Noise of energy consumption. Set to 0 if no noise
                 snn_interface=None,
                 output_layer="None",  # Use which layer of SNN to output latent features, e.g. "conv2". "None" will use
                 # raw pixels and according to the design of Stable Baselines3, the image will be processed with the
                 # Nature Atari CNN network and output a latent vector of size 256.
                 # "avg_iou" will first use a simple CNN model to predict the average IOU of the image and then RL will
                 # observe this predicted value.
                 coefficient_smoothness_penalty=30,  # Smoothness penalty: Encourages smooth changes in IOU across
                 # steps
                 coefficient_variance_penalty=200,  # Variance penalty: Penalizes high variability in IOU values
                 coefficient_range_penalty=30,  # Range penalty using IQR (IQR stands for Interquartile Range) for
                 # long sequences or max-min range for short sequences:
                 # Penalize large dispersion in IOU values
                 coefficient_energy_penalty=0.01,  # Penalize the current step's energy cost regardless of IOU
                 coefficient_special_energy_penalty=2,  # Penalize the energy cost in special case, e.g. IOU = 0
                 iou_exponent=0.3,  # Exponent for enhancing high IOU values
                 iou_base_reward_scale=30,  # Base scale factor for the reward calculation
                 concave_function1_n=2.14,  # function 1 - (1 - x) ^ n
                 reward_offset_per_step=-15,
                 iou_predict_model=None,
                 debug_info=False):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.iou_result_path = iou_result_path
        self.snn_dataset = snn_dataset
        self.image_dir = os.path.join(dataset_dir, 'images')
        self.ini_energy = ini_energy
        self.remaining_energy = ini_energy
        self.image_resolution = image_resolution
        self.images_per_group = images_per_group
        self.remaining_images = images_per_group
        self.noise_level = noise_level
        self.output_layer = output_layer
        self.snn_interface = snn_interface
        self.coefficient_smoothness_penalty = coefficient_smoothness_penalty
        self.coefficient_variance_penalty = coefficient_variance_penalty
        self.coefficient_range_penalty = coefficient_range_penalty
        self.coefficient_energy_penalty = coefficient_energy_penalty
        self.coefficient_special_energy_penalty = coefficient_special_energy_penalty
        self.iou_exponent = iou_exponent
        self.iou_base_reward_scale = iou_base_reward_scale
        self.concave_function1_n = concave_function1_n
        self.reward_offset_per_step = reward_offset_per_step
        self.iou_predict_model = iou_predict_model
        self.debug_info = debug_info
        # self.ini_image_path is only for initialization
        self.ini_group_path = [d for d in os.listdir(self.image_dir) if
                               os.path.isdir(os.path.join(self.image_dir, d)) and d != 'rest_images'][0]
        self.ini_group_path = os.path.join(self.image_dir, self.ini_group_path)
        self.ini_image_path = [f for f in os.listdir(self.ini_group_path) if f.endswith('.jpg')][0]
        self.ini_image_path = os.path.join(self.ini_group_path, self.ini_image_path)

        assert group_selection_mode in ["random", "sequential"], f"group_selection_mode must be in " \
                                                                 f"[\"random\", \"sequential\"], but " \
                                                                 f"{group_selection_mode} is given!"
        self.group_selection_mode = group_selection_mode

        assert snn_model_type in ["slim", "squeeze"], f"snn_model_type must be in [\"slim\", \"squeeze\"], but " \
                                                      f"{snn_model_type} is given!"
        self.snn_model_type = snn_model_type

        assert noise_type in ["gaussian", "uniform"], f"noise_type must be in [\"gaussian\", \"uniform\"], but " \
                                                      f"{noise_type} is given!"
        self.noise_type = noise_type

        # Define action space: 4 discrete actions
        self.action_space = spaces.Discrete(4)

        # Define observation space
        if self.output_layer == "None" or self.output_layer is None:
            # Image input: 512 x 512 x 3 pixels (0-255, uint8)
            image_space = spaces.Box(low=0, high=255, shape=(image_resolution, image_resolution, 3), dtype=np.uint8)
        elif self.output_layer == "avg_iou":
            image_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
            print(f"Will use average IOU prediction to replace image input.")
        else:
            self.latent_feature_length = self.snn_interface.infer_from_rl_latent_features(
                image_path=self.ini_image_path,
                output_layer=self.output_layer,
                snn_dataset=self.snn_dataset,
                width=1.0
            ).shape[0]
            image_space = spaces.Box(low=float('-inf'), high=float('inf'), shape=(self.latent_feature_length,),
                                     dtype=np.float32)
            # image_space = spaces.Box(low=-0.00015, high=0.0001, shape=(self.latent_feature_length,),
            #                          dtype=np.float32)
            print(f"Will use {self.snn_model_type} SNN model's {self.output_layer} layer to produce "
                  f"{self.latent_feature_length} latent features.")

        # Remaining energy: continuous space but integer values from 0 to 120
        energy_space = spaces.Box(low=0, high=ini_energy, shape=(1,), dtype=np.float32)
        # Remaining images to process: continuous space but integer values from 0 to 10
        #image_count_space = spaces.Box(low=0, high=images_per_group, shape=(1,), dtype=np.float32)
        image_count_space = spaces.Discrete(11)

        # Combine all parts of the observation space
        # self.observation_space = spaces.Dict({
        #     'image': image_space,
        #     # 'energy': energy_space,
        #     # 'image_left': image_count_space
        # })
        # self.observation_space = spaces.Discrete(11)
        # self.observation_space = spaces.Box(low=0, high=images_per_group, shape=(1,), dtype=np.float32)
        # self.observation_space = spaces.Dict({'image_left': self.observation_space})
        self.observation_space = image_space

        # Initialize environment state
        self.obs = None
        self.reward = 0
        self.current_action = None

        # Useful parameter
        self.total_steps = 0
        self.total_episodes = 0
        self.info = {}
        # If the current episode has proceeded some steps but not yet achieved a "done"
        self.episode_in_process = False
        # If the current episode has just achieved a "done"
        self.done_episode = False
        # If the environment is in evaluation
        self.in_eval = False
        # Which group of data is currently chosen
        self.current_group = None
        self.current_group_path = None
        self.current_image_path = None
        # A list contains all groups' names
        self.groups = None
        self.group_index = 0
        # A list contains all images' names in the current group
        self.image_files = []
        self.current_image_index = 0
        # Define the energy consumption for each model and action
        # Original energy consumption - Slim (SNN U-Net)
        # [0.109, 0.231, 0.49, 0.513]
        self.energy_consumption_slim = [4.72, 10.00, 21.21, 22.21]
        # Original energy consumption - Squeeze (SNN Squeeze U-Net)
        # [0.023, 0.033, 0.053, 0.094]
        self.energy_consumption_squeeze = [6.97, 10.00, 16.06, 28.48]
        # A list stores all iou scores in one episode
        self.iou_sequence = []
        self.iou_results = self.load_iou_results()
        self.width_list_str = ['0.25', '0.5', '0.75', '1.0']
        self.width_list_float = [0.25, 0.5, 0.75, 1.0]
        # Use a hash table to store latent_features to speed up
        self.latent_features_table = {}



        # Useful Info
        # Store the action and reward in evaluation
        self.eval_action_list = []
        self.eval_reward_list = []
        # self.eval_reward_mean_list = []
        # self.eval_reward_std_list = []
        # self.train_reward_mean_list = []
        # self.train_reward_std_list = []

    def step(self, action):
        self.current_action = action
        self.total_steps += 1
        self.episode_in_process = True
        self.done_episode = False
        # Update environment state based on action
        self.current_image_index += 1

        if self.snn_model_type == "slim":
            self.energy_consumed = self.add_noise(self.energy_consumption_slim[action])
        elif self.snn_model_type == "squeeze":
            self.energy_consumed = self.add_noise(self.energy_consumption_squeeze[action])
        else:
            print(f"Unexpected snn_model_type: {self.snn_model_type}")

        self.remaining_energy -= self.energy_consumed
        self.remaining_images = len(self.image_files) - self.current_image_index

        # Calculate reward
        self.reward = self.get_reward()
        if self.debug_info:
            print(f"Step {self.total_steps}, Reward = {self.reward}")

        self.obs = self._get_observation()

        # Determine if the episode is done
        # Truncated is for time-limits when time is not part of the observation space.
        # If time is part of your game, then it should be part of the observation space,
        # and the time-limit should trigger terminated, not truncated.
        terminated = self.remaining_energy <= 0 or self.remaining_images <= 0
        truncated = False

        if terminated or truncated:
            self.done_episode = True
            self.total_episodes += 1
            self.episode_in_process = False

        # self.obs = self.state
        info = {}

        return self.obs, self.reward, terminated, truncated, info

    def get_reward(self):
        if self.iou_results is not None:
            image_name = self.image_files[self.current_image_index - 1]
            image_name = os.path.splitext(os.path.basename(image_name))[0]
            image_iou_results = self.iou_results[image_name]
            # self.current_iou = image_iou_results["ious"][self.width_list_str[self.current_action]]
            iou_lst = [image_iou_results["ious"][w] for w in self.width_list_str]
            self.current_iou = self.min_max_normalize(iou_lst, self.current_action)
            # self.current_iou = 1 if iou_lst[self.current_action] == max(iou_lst) else 0
            # self.current_iou = iou_lst[self.current_action]
        else:
            self.snn_results = self.snn_interface.infer_from_rl(image_path=self.current_image_path,
                                                                snn_dataset=self.snn_dataset,
                                                                width=self.width_list_float[self.current_action])
            self.current_iou = self.snn_results["test/iou/weeds"]

        self.iou_sequence.append(self.current_iou)

        if self.current_iou == 0:
            # Special case: Incorrect image detected, IOU is 0
            # In this case, we want the agent to choose the action with the least energy consumption
            # if self.current_action == 0:
            #     reward = 0  # No energy penalty because the action with the least energy consumption is chosen
            # else:
            #     reward = -(self.coefficient_special_energy_penalty * self.energy_consumed)
            # print(f"Reward = {reward}")
            reward = 0
        else:
            # Normal case: Correct image processed, calculate reward based on IOU and stability
            # Filter out any IOU values of 0 from the sequence to avoid skewing the stability penalties
            filtered_sequence = [iou for iou in self.iou_sequence if iou > 0]

            if len(filtered_sequence) > 1:
                # Smoothness penalty: Encourages smooth changes in IOU across steps
                smoothness_penalty = self.coefficient_smoothness_penalty * sum(
                    abs(filtered_sequence[i] - filtered_sequence[i - 1]) for i in range(1, len(filtered_sequence))) / (
                                                 len(filtered_sequence) - 1)

                # Variance penalty: Penalizes high variability in IOU values
                mean_iou = sum(filtered_sequence) / len(filtered_sequence)
                variance_penalty = self.coefficient_variance_penalty * sum(
                    (x - mean_iou) ** 2 for x in filtered_sequence
                ) / len(filtered_sequence)

                # Range penalty using IQR (IQR stands for Interquartile Range) for
                # long sequences or max-min range for short sequences:
                # Penalize large dispersion in IOU values

                # Check if there are enough elements to compute IQR
                if len(filtered_sequence) >= 4:
                    # Range penalty using IQR: Penalize large dispersion in IOU values
                    q1 = np.percentile(filtered_sequence, 25)  # 25th percentile
                    q3 = np.percentile(filtered_sequence, 75)  # 75th percentile
                    iqr = q3 - q1  # Interquartile range
                    range_penalty = self.coefficient_range_penalty * iqr  # Penalize based on IQR
                else:
                    # Fallback to max-min range for short sequences
                    max_min_range = max(filtered_sequence) - min(filtered_sequence)
                    range_penalty = self.coefficient_range_penalty * max_min_range  # Penalize based on max-min range
            else:
                # No penalty if insufficient data to calculate stability metrics
                smoothness_penalty = 0
                variance_penalty = 0
                range_penalty = 0

            # Penalize the current step's energy cost regardless of IOU
            energy_penalty = self.coefficient_energy_penalty * self.energy_consumed

            # Calculate the reward with a base of high IOU encouragement, reduced by stability penalties
            # reward = self.iou_base_reward_scale * (self.concave_function1(self.current_iou, self.concave_function1_n)) \
            #          - smoothness_penalty - variance_penalty - range_penalty - energy_penalty \
            #          + self.reward_offset_per_step
            #
            # print(f"Reward = {round(self.iou_base_reward_scale * (self.concave_function1(self.current_iou, self.concave_function1_n)), 2)}"
            #       f" - {round(smoothness_penalty, 2)}"
            #       f" - {round(variance_penalty, 2)}"
            #       f" - {round(range_penalty, 2)}"
            #       f" - {round(energy_penalty, 2)}"
            #       f" + {self.reward_offset_per_step}")
            reward = 1 * self.current_iou

        return reward

    def concave_function1(self, iou, n):
        # Computes the value of the function 1 - (1 - x)^n
        return 1 - (1 - iou) ** n
        # Requirements:
        # 1. The function should be concave and increasing.
        #    - A concave function means that the second derivative is non-positive,
        #      indicating that the function curves downwards. It is important to have
        #      a concave function in this case because we want the function to show diminishing
        #      returns; that is, the impact on y should be greater when x is smaller,
        #      compared to when x is larger. Specifically, increasing X from 0.1 to 0.11
        #      should have a larger effect on Y than increasing X from 0.9 to 0.91.
        #    - The function being increasing means that its first derivative is positive,
        #      so the function value rises as x increases.
        # 2. At x = 0.65, the derivative of the function should be approximately equal to x.
        # 3. At x = 0.65, the function value should be around 0.9.
        # 4. At x = 1, the function value should be exactly 1.
        # 5. The domain of the function is [0, 1], and the range should also be [0, 1].
        # Example parameters based on requirements: n = 2.14


    def reset(self, seed=None, options=None):
        self.choose_group()
        self.image_files = sorted([f for f in os.listdir(self.current_group_path) if f.endswith('.jpg')])
        self.current_image_index = 0
        self.remaining_energy = self.ini_energy
        self.remaining_images = len(self.image_files) - self.current_image_index
        assert self.remaining_images == self.images_per_group, f"The {self.current_group} has {self.remaining_images}" \
                                                               f" images, while we expect to have " \
                                                               f"{self.images_per_group} images!"
        # Reset environment state and return initial observation
        self.iou_sequence = []
        self.obs = self._get_observation()
        self.info = {}

        return self.obs, self.info

    def choose_group(self):
        # Choose a random group
        if self.groups is None:
            self.groups = [d for d in os.listdir(self.image_dir) if
                           os.path.isdir(os.path.join(self.image_dir, d)) and d != 'rest_images']
        if self.group_selection_mode == 'random':
            self.current_group = random.choice(self.groups)
        elif self.group_selection_mode == 'sequential':
            self.current_group = self.groups[self.group_index]
            self.group_index = (self.group_index + 1) % len(self.groups)
        self.current_group_path = os.path.join(self.image_dir, self.current_group)
        if self.debug_info:
            print(f"Selected group: {self.current_group_path}")

    def _get_observation(self):
        if self.current_image_index < len(self.image_files):
            self.current_image_path = os.path.join(self.image_dir, self.current_group,
                                                   self.image_files[self.current_image_index])
            if self.debug_info:
                print(f"Selected image: {self.current_image_path}")
            if self.output_layer == "None" or self.output_layer is None:
                image = cv2.imread(self.current_image_path)
                image = cv2.resize(image, (512, 512))
            elif self.output_layer == "avg_iou":
                self.avg_iou = predict_avg_iou(image_path=self.current_image_path, iou_model=self.iou_predict_model,
                                               norm_params_path="./image_processing/norm_params_only_avg_iou.json")
                image = np.array([np.float32(self.avg_iou)], dtype=np.float32)
            else:
                if self.current_image_path in self.latent_features_table.keys():
                    image = self.latent_features_table[self.current_image_path]
                else:
                    image = self.snn_interface.infer_from_rl_latent_features(image_path=self.current_image_path,
                                                                             output_layer=self.output_layer,
                                                                             snn_dataset=self.snn_dataset,
                                                                             width=1.0)
                    self.latent_features_table[self.current_image_path] = image
                    print(f"Add {self.current_image_path} to latent features hash table.")
                    print(np.max(image), np.min(image))
        else:  # Return 0 as observation because no image left
            if self.output_layer == "None" or self.output_layer is None:
                image = np.zeros((512, 512, 3), dtype=np.uint8)
            elif self.output_layer == "avg_iou":
                image = np.zeros((1,), dtype=np.float32)
            else:
                image = np.zeros((self.latent_feature_length,), dtype=np.float32)

        # obs = {
        #     'image': image,
        #     # 'energy': np.array([self.remaining_energy], dtype=np.float32),
        #     # #'image_left': np.array([len(self.image_files) - self.current_image_index], dtype=np.float32)
        #     # 'image_left': len(self.image_files) - self.current_image_index
        # }
        # obs = len(self.image_files) - self.current_image_index
        # obs = np.array([len(self.image_files) - self.current_image_index], dtype=np.float32)
        # obs = {'image_left': obs}
        obs = image
        return obs

    def set_snn_interface(self, snn_interface):
        self.snn_interface = snn_interface

    def add_noise(self, value):
        if self.noise_type == 'gaussian':
            noise = np.random.normal(0, self.noise_level * value)
        elif self.noise_type == 'uniform':
            noise = np.random.uniform(-self.noise_level * value, self.noise_level * value)
        else:
            noise = 0
        return value + noise

    def load_iou_results(self):
        if self.iou_result_path is None or self.iou_result_path == "None":
            print(f"IOU results json file is not provided.")
            return None
        with open(self.iou_result_path, 'r') as f:
            iou_results = json.load(f)
        return {item['image']: item for item in iou_results}

    def min_max_normalize(self, lst, index):
        """
        Normalize the input list using min-max normalization and return the value at the specified index.

        Parameters:
        lst (list): The list to be normalized
        index (int): The index of the value to be retrieved after normalization

        Returns:
        float: The normalized value at the specified index
        """
        if not lst:
            raise ValueError("The input list cannot be empty")
        if index < 0 or index >= len(lst):
            raise IndexError("Index out of range")

        min_val = min(lst)
        max_val = max(lst)

        # If all elements are the same, min and max are equal, normalization is not meaningful, return 1
        if min_val == max_val:
            print(f"min and max elements are equal, {min_val}")
            return 1

        normalized_value = (lst[index] - min_val) / (max_val - min_val)
        return normalized_value





