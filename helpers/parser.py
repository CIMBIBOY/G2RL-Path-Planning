from distutils.util import strtobool
import argparse
    
def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_name', type=str, default='train', 
        help='Specifiy the name of the current train.')
    parser.add_argument('--seed', type=int, default=42, 
        help='Random seed for reproducibility.')
   
    parser.add_argument("--eval", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, runtime is performance evaluation from existing model weights.")
    parser.add_argument('--eval_steps', type=int, default=100, 
        help='Define the number of timesteps the agent will be evaluated for.')
    
    parser.add_argument('--method', type=str, choices=['dqn', 'qnet', 'mppo'], default='dqn',
        help='Choose the training method: deep Q-network, traditional Q-network or masked PPO agent')
    parser.add_argument('--train', type=str, choices=['scratch', 'retrain'], default='scratch',
        help='Choose whether to train from scratch or retrain an existing model.')
    parser.add_argument('--model_weights', type=str, default=None,
        help='Path to the model weights file (required if --train is set to retrain).')
    
    parser.add_argument("--num_envs", type=int, default=4,
        help="the number of environments running in parallel")
    parser.add_argument("--time_dim", type=int, default=4,
        help="the number of time dimensions used by the convolutional layers")
    parser.add_argument("--learning_rate", type=float, default=3.5e-5,
        help="the learning rate of the optimizer")
    parser.add_argument("--total_timesteps", type=int, default=128000,
        help="total timesteps of the experiments")
    
    parser.add_argument("--torch_deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb_project_name", type=str, default="G2RL",
        help="the wandb's project name")
    parser.add_argument("--wandb_entity", type=str, default='czimbermark',
        help="the entity (team) of wandb's project")

    parser.add_argument("--capture_video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="weather to capture videos of the agent performances (check out `training_images` folder)")
    parser.add_argument("--pygame", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="weather to capture pygame rendering of the agent performances")
    parser.add_argument('--cmd_log', type=int, default=5, 
                help='Set command line log frequency')

    # PPO algorithm specific arguments
    parser.add_argument("--num_steps", type=int, default=128,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal_lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gae", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Use GAE for advantage computation")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae_lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num_minibatches", type=int, default=4,
        help="the number of mini_batches")
    parser.add_argument("--update_epochs", type=int, default=4,
        help="the K epochs to update the policy")
    parser.add_argument("--norm_adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip_coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip_vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent_coef", type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument("--vf_coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max_grad_norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target_kl", type=float, default=None,
        help="the target KL divergence threshold")
    
    # Q-Learning 
    parser.add_argument('--explore', type=int, default=200000, 
        help='Set exploration steps for e-greedy decay for q-network')
    parser.add_argument('--batch', type=int, default=32, 
        help='Set batch for q-network')
    
    args = parser.parse_args()

    args.batch_size = int(args.num_envs * args.num_steps)

    # fmt: on
    return args