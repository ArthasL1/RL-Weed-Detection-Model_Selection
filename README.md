# Reinforcement Learning for Weed Detection Model Selection

## Project Overview
Food security is a critical issue that humanity faces, and it is important that crops’ yield is maximized. Weeds can significantly reduce the amount of food we harvest, thus their detection remains crucial. This task can be performed by drones that capture images of land areas, but the on-device nature of the problem brings significant challenges. Researchers in approximate on-device computing have developed techniques that can dynamically switch between a collection of deep neural network (DNN) models, each with a different size, average inference time, and accuracy.

Deploying this method for weed detection on images of different complexity taken from a drone camera can potentially save energy and increase the drone’s flight time, as a smaller network uses less energy than the full network. However, deciding which DNN variant to use during the flight is challenging - if we use a small network for complex images, we might fail to detect weeds; if we use a full network on a trivial image, we miss an opportunity to save energy.

In this project, we investigate the applicability of reinforcement learning (RL) to control DNN adaptation. Given indicators of environmental conditions (image complexity, battery level, etc.), the task is to train an RL agent to select among several pre-trained DNN models. The reward-driven, trial-and-error paradigm of RL has the potential to discover model selection strategies that perform better than current methods.

## Repository Structure

- `env.py`: Defines the custom environment `WeedDetectionEnv` for training the RL agent.
- `config.py`: Handles argument parsing for configuring training parameters.
- `train_RL.py`: Contains the main training loop for the RL agent, evaluation functions, and callback definitions.

## Environment Details (`env.py`)

### WeedDetectionEnv Class
This class simulates the drone's weed detection task, including:

- **Action Space**: Four discrete actions.
- **Observation Space**: 
  - Image input (512 x 512 x 3 pixels).
  - Remaining energy.
  - Remaining images to process (0 to 10).

### Methods
- `__init__()`: Initializes the environment.
- `step(action)`: Updates the environment state based on the agent's action, calculates rewards, and checks if the episode is done.
- `reset(seed=None, options=None)`: Resets the environment state.

## Configuration Details (`config.py`)

### parse_arguments Function
Parses command-line arguments for configuring the training process, including:
- `--model_name`: Name of the model.
- `--model_type`: Type of the RL model (e.g., PPO).
- `--learn_timesteps`: Total steps to train the RL agent.
- `--RL_n_steps`: Number of steps to run for each environment per update.

## Training Script (`train_RL.py`)

### Training Process
The script defines and trains the RL model using Stable Baselines3. The key components include:

- **RewardEvalCallback Class**: Custom callback for evaluating the model during training.
- **eval_RL Function**: Evaluates the RL model and calculates the mean and standard deviation of rewards and episode lengths.
- **train_RL Function**: Main function to define the RL model, train it, evaluate it, and save results.

### Running the Training
To train the RL model, run the following command:

```bash
python train_RL.py --model_name PPO_test0 --model_type PPO --learn_timesteps 2560 --RL_n_steps 256
```

### Output
The training script will output:
- Trained model files saved in the specified directory.
- Learning and evaluation plots saved as images.



