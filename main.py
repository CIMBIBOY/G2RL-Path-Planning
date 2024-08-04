import argparse
from WarehouseEnv import WarehouseEnvironment
from train_utils import set_seed, print_cuda_info
from dqn_network import dqn_training
from q_learning import q_learning_training
from parser_init import init_parser

'''
python3 main.py --render off --method dqn --epochs 100000 --timesteps 33 --metal cuda --train scratch --batch 512 --train_num 2

python3 main.py --render off --method dqn --epochs 100000 --timesteps 33 --metal cuda --train retrain --model_weights G2RL-Path-Planning/weights/dqn_model_cpu.pth --batch 512

--model_weights models/dqn_model.pth
'''

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parser for training script of G2RL.')
    init_parser(parser)
    args = parser.parse_args()

    # Set the seed for reproducibility
    set_seed(args.seed)

    num_ep = args.epochs
    num_timesteps = args.timesteps
    batch = args.batch
    train_num = args.train_num

    if args.render == 'all': video = True
    else: video = False

    if args.render == 'pygame':
        env = WarehouseEnvironment(pygame_render=True)
        _, state = env.reset() # image of first reset
        print(f"Input tensor dimension (state.shape): {state.shape}")
    elif args.render == 'off':
        env = WarehouseEnvironment(pygame_render=False)
        _, state = env.reset() # image of first reset
        print(f"Input tensor dimension (state.shape): {state.shape}")
    else:
        print("Render automatically set to False!")
        env = WarehouseEnvironment(pygame_render=False)
        _, state = env.reset() # image of first reset
        print(f"Input tensor dimension (state.shape): {state.shape}")

    if args.metal == 'cuda':
        metal = 'cuda'
        print_cuda_info()  # Print CUDA info if CUDA is selected
    elif args.metal == 'cpu':
        metal = 'cpu'
        print('Metal is CPU')
    else: print('Only cpu or cuda accelaration is supported')

    model_weights_path = None
    if args.train == 'retrain':
        if args.model_weights is None:
            parser.error("The --model_weights argument is required when --train is set to 'retrain'.")
        model_weights_path = args.model_weights

    if args.method == 'dqn':
        dqn_training(env, num_episodes = num_ep, timesteps_per_episode=num_timesteps, save_images = video, batch_size=batch, train_num=train_num)
    elif args.method == 'qnet':
        q_learning_training(env, num_episodes = num_ep, timesteps_per_episode=num_timesteps, save_images = video)   
    else: print("No method choosen or type error in parsing argument! Please eaither use command: \npython main.py --method dqn \nor\n python main.py --method qnet")

    env.close()

    