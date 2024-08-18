
from WarehouseEnv import WarehouseEnvironment
from train_utils import set_seed, print_cuda_info
from dqn_network import dqn_training
from q_learning import q_learning_training
from ppo_network import ppo_training
from parser import parse_args
import time
import pygame
from torch.utils.tensorboard import SummaryWriter
import torch

'''
python3 main.py --train_name czm1 --seed 42 --method dqn --train scratch --total_timesteps 100001 --num_steps 1000 --pygame --cmd_log 5 --explore 20000

python3 main.py --train_name czm1 --seed 31 --method mppo --train scratch --total_timesteps 100000 --num_steps 256 --cmd_log 5 --learning_rate 3e-5

python3 main.py --train_name czm1 --seed 37 --method dqn --train scratch --total_timesteps 100000 --num_steps 1000 --pygame --cmd_log 5 --batch 64 --explore 20000

'''

if __name__ == '__main__':
    args = parse_args()
    run_name = f"{args.train_name}_{args.method}_{args.seed}_{int(time.time())}"
 
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )

    set_seed(args.seed, args.torch_deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print_cuda_info()

    pygame.init()
    time.sleep(1)
    
    # Creating env
    env = WarehouseEnvironment(pygame_render=args.pygame)
    _, state = env.reset()
    print(f"Input tensor dimension (state.shape): {state.shape}")

    if env.pygame_render:
        env.render()  

    model_weights_path = None
    if args.train == 'retrain':
        if args.model_weights is None:
            args.error("The --model_weights argument is required when --train is set to 'retrain'.")
        model_weights_path = args.model_weights

    if args.method == 'dqn':
        dqn_training(env, args.total_timesteps, args.num_steps, args.capture_video, model_weights_path=model_weights_path, batch_size=args.batch, train_name=run_name, cmd_log=args.cmd_log, explore=args.explore)
    elif args.method == 'qnet':
        q_learning_training(env, args.total_timesteps, args.num_steps, args.capture_video)   
    if args.method == 'mppo':
        agent = ppo_training(env, args)  
    else: print("No method choosen or type error in parsing argument! Please eaither use command: \npython main.py --method dqn \nor\n python main.py --method qnet")

    env.close()

    