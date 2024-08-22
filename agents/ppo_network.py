# ppo_network.py

import torch
import numpy as np
from agents.ppo_agent import PPOAgent
from agents.cnn_for_ppo import CNNLSTM
from eval.eval import evaluate_performance
import time
from helpers.model_summary import print_model_summary_ppo
from environment.WarehouseEnv import WarehouseEnvironment

def ppo_training(env, args, train_name, writer, wandb):
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    model = CNNLSTM(args.time_dim).to(device)
    agent = PPOAgent(env, model, args, train_name, writer, wandb)

    if args.model_weights:
        try:
            agent.load(args.model_weights)
            print(f"Loaded model weights from: {args.model_weights}")
            time.sleep(2)
        except Exception as e:
            print(f"Error loading model weights: {e}")
            time.sleep(2)
    
    # Define the input size
    input_size = (args.batch_size, args.num_envs, args.time_dim, 30, 30, 4)  # (num_envs, batch_size, time_dim, height, width, channels)

    # Call the print_model_summary_ppo function once before starting the training
    print_model_summary_ppo(model, input_size, env, device)

    print(f"\nTotal number of training updates: {int(args.total_timesteps // args.batch_size)}")
    print(" ----------------- Training Started -----------------")

    pg_loss, v_loss, entropy_loss, old_approx_kl, approx_kl = agent.learn()

    print(f"Final Policy Loss: {pg_loss:.4f}")
    print(f"Final Value Loss: {v_loss:.4f}")
    print(f"Final Entropy: {entropy_loss:.4f}")
    print(f"Final KL Divergence: {approx_kl:.4f}")
    agent.save(f'./eval/weights/{train_name}.pth')

    print(" ---------- Training Finished ----------")

    # Final evaluation
    print(" ---------- Final Evaluation ----------")
    final_performance = evaluate_performance(env.envs[0], args, num_episodes=100, agent = agent)
    print(f"Final average reward: {final_performance['avg_reward']:.2f}")
    print(f"Final average moving cost: {final_performance['moving_cost']:.4f}")
    print(f"Final average detour percentage: {final_performance['detour_percentage']:.2f}%")
    print(f"Final average computing time: {final_performance['computing_time']:.4f} s/step")
    print(f"Total failed paths: {final_performance['failed_paths']}")
    print(f"Total goals reached: {final_performance['agent_reached_goal']}")

    return agent
