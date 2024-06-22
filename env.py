import numpy as np
import pdb
import gymnasium as gym
from stable_baselines3 import PPO
import random
import torch
from gymnasium import spaces

# SEED = 2024
SEED = None

DEBUG_INFO = False
if SEED is not None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

class WeedDetectionEnv(gym.Env):
    def __init__(self, ):
        super().__init__()
        # Define action space: 4 discrete actions
        self.action_space = spaces.Discrete(4)

        # Define observation space
        # Image input: 640 x 640 x 3 pixels (0-255, uint8)
        image_space = spaces.Box(low=0, high=255, shape=(640, 640, 3), dtype=np.uint8)
        # Remaining energy: continuous space but integer values from 0 to 100
        energy_space = spaces.Box(low=0, high=120, shape=(1,), dtype=np.float32)
        # Remaining images to process: continuous space but integer values from 0 to 10
        image_count_space = spaces.Box(low=0, high=10, shape=(1,), dtype=np.float32)

        # Combine all parts of the observation space
        self.observation_space = spaces.Dict({
            'image': image_space,
            'energy': energy_space,
            'image_left': image_count_space
        })

        # Initialize environment state, but this will be overwritten in reset()
        self.ini_energy = 120
        self.ini_image_left = 10
        self.state = {
            'image': np.random.randint(0, 256, (640, 640, 3), dtype=np.uint8),
            'energy': np.array([self.ini_energy], dtype=np.float32),
            'image_left': np.array([self.ini_image_left], dtype=np.float32)
        }
        self.obs = self.state
        self.reward = 0

        # Useful parameter
        self.total_steps = 0
        self.total_episodes = 0
        self.info = {}
        self.episode_in_process = False  # If the current episode has proceeded some steps but not yet achieved a "done"
        self.done_episode = False  # If the current episode has just achieved a "done"
        self.in_eval = False  # If the environment is in evaluation

    def step(self, action):
        self.total_steps += 1
        self.episode_in_process = True
        self.done_episode = False
        # Update environment state based on action
        self.state['image'] = np.random.randint(0, 256, (640, 640, 3), dtype=np.uint8)
        self.state['energy'] = np.array([max(0, self.state['energy'][0] - (action + 1)*10)], dtype=np.float32)
        self.state['image_left'] = np.array([max(0, self.state['image_left'][0] - 1)], dtype=np.float32)

        # Calculate reward
        self.reward = 5 + 5 * (action + 1)

        # Determine if the episode is done
        # Truncated is for time-limits when time is not part of the observation space.
        # If time is part of your game, then it should be part of the observation space,
        # and the time-limit should trigger terminated, not truncated.
        terminated = self.state['energy'] == 0 or self.state['image_left'] == 0
        truncated = False

        if terminated or truncated:
            self.done_episode = True
            self.total_episodes += 1
            self.episode_in_process = False

        self.obs = self.state
        info = {}

        return self.obs, self.reward, terminated, truncated, info


    def reset(self, seed=None, options=None):
        # Reset environment state and return initial observation
        self.state = {
            'image': np.random.randint(0, 256, (640, 640, 3), dtype=np.uint8),
            'energy': np.array([self.ini_energy], dtype=np.float32),
            'image_left': np.array([self.ini_image_left], dtype=np.float32)
        }
        self.obs = self.state
        self.info = {}

        return self.obs, self.info
