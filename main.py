
from matplotlib import MatplotlibDeprecationWarning
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
python3 main.py --train_name CS47 --seed 515 --method mppo --train scratch --total_timesteps 10240000 --num_steps 256 --cmd_log 5 --num_envs 4 --time_dim 13 --learning_rate 2e-8 --pygame

python3 main.py --train_name TS_4_7_noBD --cuda --seed 540 --method mppo --train scratch --total_timesteps 30720000 --num_steps 512 --cmd_log 5 --learning_rate 3e-5 --num_envs 3 --num_minibatches 3 --update_epochs 3 --time_dim 7 --track 

python3 main.py --train_name titanS --cuda --seed 437 --method mppo --train retrain --model_weights eval/weights/TS47noBD_mppo_525_1724684017.pth --total_timesteps 10240000 --num_steps 1024 --cmd_log 5 --learning_rate 1e-5 --num_envs 4 --track --clip_coef 0.1 --max_grad_norm 0.4 --pygame

python3 main.py --train_name Q7 --seed 37 --method dqn --train scratch --total_timesteps 100000 --num_steps 1000 --cmd_log 5 --batch 64 --explore 200000

python3 main.py --train_name eval --seed 900 --method mppo --train retrain --model_weights eval/weights/TS_4_7_noBD_mppo_540_1725258796.pth --eval --eval_steps 100 --pygame

'''

if __name__ == '__main__':
    args = parse_args()
    run_name = f"{args.train_name}_{args.method}_{args.seed}_{int(time.time())}"

    # warnings.filterwarnings("error")
    # warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)
 
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True
        )
        print("\nWandb initialized successfully.")  # Test if this is captured in Wandb logs
    else:
        wandb = None
        print("Wandb not initialized.")  # Test if this prints to stdout

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
    
    if not args.eval and args.method == 'mppo':
        # Set up the parallel environments
        envs = gym.vector.SyncVectorEnv(
            [make_custom_env(seed=args.seed, idx=i, height=48, width=48, amr_count=2, max_amr=25, time_dimension=args.time_dim, pygame_render=args.pygame) for i in range(args.num_envs)]
        )
        # Reset num_envs number environment
        state, info = envs.reset()
        print("\nFor input shape expecting: (num_envs, batch_size, time_dim, obs_width, obs_height, chanels)")
        print(f"State tensor (observation) dimension: {state.shape}\n")

        # Setting long horizon for ppo
        for i in range(args.num_envs):
            envs.envs[i].horizon = 'long'
            envs.envs[i].max_step = 4096

        # Render the first env instance
        if args.pygame:
            envs.envs[0].render() 

    elif args.eval: 
        envs = gym.vector.SyncVectorEnv(
            [make_custom_env(seed=args.seed, idx=i, height=48, width=48, amr_count=25, max_amr=25, time_dimension=args.time_dim, pygame_render=args.pygame) for i in range(1)]
        )
        # Reset single number environment
        state, info = envs.reset()
        print("\nFor input shape expecting: (num_envs, batch_size, time_dim, obs_width, obs_height, chanels)")
        print(f"State tensor (observation) dimension: {state.shape}\n")

        # Render the env
        if args.pygame:
            envs.envs[0].render()  
    
    elif args.method != 'mppo': 
        if args.eval: args.seed += 1
        # Create single env
        env = WarehouseEnvironment(time_dimension=args.time_dim, pygame_render=args.pygame, seed = args.seed)
        # Reset single number environment
        state, info = env.reset()
        print("\nFor input shape expecting: (num_envs, batch_size, time_dim, obs_width, obs_height, chanels)")
        print(f"State tensor (observation) dimension: {state.shape}\n")
        
        # Render the env
        if args.pygame:
            env.render() 

    model_weights_path = None
    if args.train == 'retrain':
        if args.model_weights is None:
            args.error("The --model_weights argument is required when --train is set to 'retrain'.")
        model_weights_path = args.model_weights

    if args.eval and model_weights_path is not None:
        final_performance = evaluate_performance(envs, args, args.eval_steps, train_name=run_name, agent = None, wandb=wandb)
        if final_performance == 0: 
            print(f"-------------------------------------------------------------------")
            print("Eval was unsuccesful, possible problem with model_weights argument!")
            print(f"-------------------------------------------------------------------")
        else: 
            print(f"-------------------------------------------------------------------")
            print(f"Final average reward: {final_performance['avg_reward']}")
            print(f"Final average moving cost: {final_performance['moving_cost']:.4f}")
            print(f"Final average detour percentage: {final_performance['detour_percentage']:.2f}%")
            print(f"Final average computing time: {final_performance['computing_time']:.4f} s/step")
            print(f"Total failed paths: {final_performance['failed_paths']}")
            print(f"Total goals reached: {final_performance['agent_reached_goal']}")
            print(f"Total max steps reached: {final_performance['max_steps_reached']}")
            print(f"Total lost global guidance: {final_performance['no_global_guidance']}")
            print(f"Total collisions with obstacles: {final_performance['collisions_with_obstacles']}")
            print(f"-------------------------------------------------------------------")

    elif args.eval is False: 
        if args.method == 'dqn':
            dqn_training(env, args.total_timesteps, args.num_steps, args.capture_video, model_weights_path=model_weights_path, batch_size=args.batch, train_name=run_name, cmd_log=args.cmd_log, explore=args.explore)
        elif args.method == 'qnet':
            q_learning_training(env, args.total_timesteps, args.num_steps, args.capture_video)   
        if args.method == 'mppo':
            agent = ppo_training(envs, args, run_name, writer, wandb=wandb)  
    else: print("No method choosen or type error in parsing argument! Please use command like:\npython3 main.py --train_name czm1 --seed 31 --method mppo --train scratch --total_timesteps 100000 --num_steps 256 --cmd_log 5 --learning_rate 3e-5\nOr use --eval flag for evaluation, which requires the specification of model_weights")

    envs.close()
    if args.track:
        writer.close()

    