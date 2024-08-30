import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', type=str, default='PPO_squeeze_1024batch_valid_minmax_7group_bestOnly_imageOnly_gamma099_ent000_RLlr000003_MLP_fire08_normalizedIOU_test0',
                        help='The name of the model, determine the name of the folder.')
    parser.add_argument('--snn_model_type', type=str, default='squeeze',
                        help='The type of the SNN model, such as "slim", "squeeze".')
    parser.add_argument('--snn_dataset', type=str, default='geok',
                        help='The name of the SNN dataset, such as "geok", "tobacco".')
    parser.add_argument('--RL_model_type', type=str, default='PPO', help='The type of the RL model, such as PPO.')
    parser.add_argument('--output_layer', type=str, default='fire08',
                        help='Decide the space of image inputs. Use which layer of SNN to output latent features, '
                             'e.g. "conv2". "None" will use raw pixels and according to the design of Stable '
                             'Baselines3, the image will be processed with the Nature Atari CNN network and output a '
                             'latent vector of size 256. "avg_iou" will first use a simple CNN model to predict the '
                             'average IOU of the image and then RL will observe this predicted value.')
    parser.add_argument('--learn_timesteps', type=int, default=500000, help='Total steps to train the RL agent.')
    parser.add_argument('--RL_n_steps', type=int, default=1024,
                        help='For RL, the number of steps to run for each environment per update.')
    parser.add_argument('--minibatch_size', type=int, default=64,
                        help='Mini-batch size is the number of data samples used in one gradient update, '
                             'and it divides the total data collected during "n_steps" into smaller batches '
                             'for training.')
    parser.add_argument('--eval_freq', type=int, default=2048,
                        help='For RL, the number of steps to run per evaluation. '
                             'It is recommended that this value be set as an integer multiple of "RL_n_steps".')
    parser.add_argument('--eval_episodes', type=int, default=7,
                        help='For RL, the number of episodes to evaluate the policy. It is recommended that this '
                             'value be set as an integer multiple of the number groups in validation set.')
    parser.add_argument('--ent_coef', type=float, default=0.00,
                        help='For RL, entropy coefficient for the loss calculation.')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='For RL, discount factor.')
    parser.add_argument('--RL_learning_rate', type=float, default=0.00003,
                        help='For RL, learning rate.')


    return parser.parse_args()
'''
defualtCNN (Atari CNN photo => 256 vector)
Latent feature from squeeze model => 196608 latent features
state(one photo, energy, images_left) => action (0.25, 0.50, 0.75, 1.00) => reward (IOU) => A = Q - V
maximize episode reward
'''

'''
policy network
A = Q(s,a) - V(s) How good an action is compared to the whole state
loss1 = - E(min(rate (new policy / old policy) * A, clip))

value network
loss2 = V - V_target

total = loss1 + loss2 + entropy_loss (0)



'''
