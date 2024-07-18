import numpy as np
import pdb
import gymnasium as gym
from stable_baselines3 import PPO
import random
import torch
from gymnasium import spaces
import os
import cv2

# SEED = 2024
SEED = None

if SEED is not None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

class WeedDetectionEnv(gym.Env):
    def __init__(self,
                 dataset_dir="./image_data/geok_grouped/train",
                 ini_energy=120,
                 images_per_group=10,
                 image_resolution=512,
                 group_selection_mode="random",  # ["random", "sequential"]
                 ssn_model_type="slim",  # ["slim", "squeeze"]
                 noise_type="gaussian",  # ["gaussian", "uniform"]
                 noise_level=0.1,  # Set to 0 if no noise
                 debug_info=False):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.image_dir = os.path.join(dataset_dir, 'images')
        self.ini_energy = ini_energy
        self.remaining_energy = ini_energy
        self.image_resolution = image_resolution
        self.images_per_group = images_per_group
        self.remaining_images = images_per_group
        self.noise_level = noise_level
        self.debug_info = debug_info

        assert group_selection_mode in ["random", "sequential"], f"group_selection_mode must be in " \
                                                                 f"[\"random\", \"sequential\"], but " \
                                                                 f"{group_selection_mode} is given!"
        self.group_selection_mode = group_selection_mode

        assert ssn_model_type in ["slim", "squeeze"], f"ssn_model_type must be in [\"slim\", \"squeeze\"], but " \
                                                      f"{ssn_model_type} is given!"
        self.snn_model_type = ssn_model_type

        assert noise_type in ["gaussian", "uniform"], f"noise_type must be in [\"gaussian\", \"uniform\"], but " \
                                                      f"{noise_type} is given!"
        self.noise_type = noise_type

        # Define action space: 4 discrete actions
        self.action_space = spaces.Discrete(4)

        # Define observation space
        # Image input: 512 x 512 x 3 pixels (0-255, uint8)
        image_space = spaces.Box(low=0, high=255, shape=(image_resolution, image_resolution, 3), dtype=np.uint8)
        # Remaining energy: continuous space but integer values from 0 to 120
        energy_space = spaces.Box(low=0, high=ini_energy, shape=(1,), dtype=np.float32)
        # Remaining images to process: continuous space but integer values from 0 to 10
        image_count_space = spaces.Box(low=0, high=images_per_group, shape=(1,), dtype=np.float32)

        # Combine all parts of the observation space
        self.observation_space = spaces.Dict({
            'image': image_space,
            'energy': energy_space,
            'image_left': image_count_space
        })

        # Initialize environment state, but this will be overwritten in reset()
        self.obs = None
        self.reward = 0

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
        # The SNN model (interface)
        self.snn_interface = None
        # Define the energy consumption for each model and action
        # Original energy consumption - Slim (SSN U-Net)
        # [0.109, 0.231, 0.49, 0.513]
        self.energy_consumption_slim = [4.72, 10.00, 21.21, 22.21]
        # Original energy consumption - Squeeze (SSN Squeeze U-Net)
        # [0.023, 0.033, 0.053, 0.094]
        self.energy_consumption_squeeze = [6.97, 10.00, 16.06, 28.48]



    def step(self, action):
        self.total_steps += 1
        self.episode_in_process = True
        self.done_episode = False
        # Update environment state based on action
        self.current_image_index += 1

        if self.snn_model_type == "slim":
            energy_consumed = self.add_noise(self.energy_consumption_slim[action])
        elif self.snn_model_type == "squeeze":
            energy_consumed = self.add_noise(self.energy_consumption_squeeze[action])
        else:
            print(f"Unexpected snn_model_type: {self.snn_model_type}")

        self.remaining_energy -= energy_consumed
        self.remaining_images = len(self.image_files) - self.current_image_index

        # Calculate reward
        if self.snn_interface is None:
            self.reward = 10 + 5 * (action + 1)
            print("No SNN model is set up. Dummy rewards will be given!")
        else:
            if action == 0:
                snn_results = self.snn_interface.infer_from_rl(image_path=self.current_image_path, width=0.25)
            elif action == 1:
                snn_results = self.snn_interface.infer_from_rl(image_path=self.current_image_path, width=0.50)
            elif action == 2:
                snn_results = self.snn_interface.infer_from_rl(image_path=self.current_image_path, width=0.75)
            elif action == 3:
                snn_results = self.snn_interface.infer_from_rl(image_path=self.current_image_path, width=1.00)
            else:
                print(f"Unexpected action: {action}")
            self.reward = 100 * snn_results["test/iou/weeds"]
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
            image = cv2.imread(self.current_image_path)
            image = cv2.resize(image, (512, 512))
        else:
            image = np.zeros((512, 512, 3), dtype=np.uint8)

        obs = {
            'image': image,
            'energy': np.array([self.remaining_energy], dtype=np.float32),
            'image_left': np.array([len(self.image_files) - self.current_image_index], dtype=np.float32)
        }
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
