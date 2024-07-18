import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', type=str, default='PPO_slim_geok_test1', help='The name of the model, determine the '
                                                                            'name of the folder.')
    parser.add_argument('--model_type', type=str, default='PPO', help='The type of the RL model, such as PPO.')
    parser.add_argument('--learn_timesteps', type=int, default=5120, help='Total steps to train the RL agent.')
    parser.add_argument('--RL_n_steps', type=int, default=128, help='For RL, the number of steps to run for each '
                                                                     'environment per update.')
    return parser.parse_args()
