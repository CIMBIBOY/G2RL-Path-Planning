
from environment.WarehouseEnv import WarehouseEnvironment
from environment.vector_env import make_custom_env
from helpers.utils import set_seed, print_cuda_info
from agents.dqn_network import dqn_training
from agents.q_learning import q_learning_training
from agents.ppo_network import ppo_training
from helpers.parser import parse_args
import time
import pygame
from torch.utils.tensorboard import SummaryWriter
import torch
from eval.eval import evaluate_performance
import gym

'''
python3 main.py --train_name cimbi --cuda --seed 160 --method mppo --train scratch --total_timesteps 12800000 --num_steps 2048 --cmd_log 10 --learning_rate 3e-5 --num_envs 4 

python3 main.py --train_name czm1 --seed 31 --method mppo --train retrain --model_weights eval/weights/czm1_mppo_31_1724005169.pth --total_timesteps 1280 --num_steps 128 --cmd_log 5 --learning_rate 3e-5 

python3 main.py --train_name czm1 --seed 37 --method dqn --train scratch --total_timesteps 100000 --num_steps 1000 --pygame --cmd_log 5 --batch 64 --explore 20000

python3 main.py --train_name czm1 --seed 100 --method mppo --train retrain --model_weights eval/weights/czm1_mppo_31_1724005169.pth --eval --eval_steps 100

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
    
    if args.method == 'mppo':
        # Set up the parallel environments
        envs = gym.vector.SyncVectorEnv(
            [make_custom_env(seed=args.seed, idx=i, height=48, width=48, amr_count=2, pygame_render=args.pygame) for i in range(args.num_envs)]
        )
        # Reset num_envs number environment
        state, info = envs.reset()
        print(f"Input tensor dimension (state.shape): {state.shape}")

        # Render the first env instance
        if args.pygame:
            envs.envs[0].render() 

    elif args.eval or args.method != 'mppo': 
        if args.eval: args.seed += 1
        # Create single env
        env = WarehouseEnvironment(pygame_render=args.pygame, seed = args.seed)
        # Reset single number environment
        state, info = env.reset()
        print(f"Input tensor dimension (state.shape): {state.shape}")

        # Render the env
        if args.pygame:
            env.render() 

    model_weights_path = None
    if args.train == 'retrain':
        if args.model_weights is None:
            args.error("The --model_weights argument is required when --train is set to 'retrain'.")
        model_weights_path = args.model_weights

    if args.eval and model_weights_path is not None:
        final_performance = evaluate_performance(env, args, run_name, args.eval_steps)
        print(f"Final average reward: {final_performance['avg_reward']:.2f}")
        print(f"Final average moving cost: {final_performance['moving_cost']:.4f}")
        print(f"Final average detour percentage: {final_performance['detour_percentage']:.2f}%")
        print(f"Final average computing time: {final_performance['computing_time']:.4f} s/step")
        print(f"Total failed paths: {final_performance['failed_paths']}")
        print(f"Total goals reached: {final_performance['agent_reached_goal']:.0f}")
        print(f"Total max steps reached: {final_performance['max_steps_reached']:.0f}")
        print(f"Total lost global guidance: {final_performance['no_global_guidance']:.0f}")
        print(f"Total collisions with obstacles: {final_performance['collisions_with_obstacles']:.0f}")

    elif args.eval is False: 
        if args.method == 'dqn':
            dqn_training(env, args.total_timesteps, args.num_steps, args.capture_video, model_weights_path=model_weights_path, batch_size=args.batch, train_name=run_name, cmd_log=args.cmd_log, explore=args.explore)
        elif args.method == 'qnet':
            q_learning_training(env, args.total_timesteps, args.num_steps, args.capture_video)   
        if args.method == 'mppo':
            agent = ppo_training(envs, args, run_name)  
    else: print("No method choosen or type error in parsing argument! Please use command like:\npython3 main.py --train_name czm1 --seed 31 --method mppo --train scratch --total_timesteps 100000 --num_steps 256 --cmd_log 5 --learning_rate 3e-5\nOr use --eval flag for evaluation, which requires the specification of model_weights")

    envs.close()
    if args.track:
        writer.close()

    