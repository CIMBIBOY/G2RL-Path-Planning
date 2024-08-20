# ppo_network.py

import torch
import numpy as np
from agents.ppo_agent import PPOAgent
from agents.cnn_for_ppo import CNNLSTM
from eval.eval import evaluate_performance
import time
from helpers.model_summary import print_model_summary_ppo

def ppo_training(env, args, train_name, writer, wandb):
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Currently running training on device: {device}")

    model = CNNLSTM().to(device)
    agent = PPOAgent(env, model, args, train_name, writer, wandb)

    if args.model_weights:
        try:
            agent.load(args.model_weights)
            print(f"Loaded model weights from: {args.model_weights}")
            time.sleep(2)
        except Exception as e:
            print(f"Error loading model weights: {e}")
            time.sleep(2)
    
    # print_model_summary_ppo(model, (args.batch_size, 4, 1, 30, 30, 4), args.batch_size, env, device)

    print(" ---------- Training Started ----------")

    pg_loss, v_loss, entropy_loss, old_approx_kl, approx_kl = agent.learn()

    print(f"Final Policy Loss: {pg_loss:.4f}")
    print(f"Final Value Loss: {v_loss:.4f}")
    print(f"Final Entropy: {entropy_loss:.4f}")
    print(f"Final KL Divergence: {approx_kl:.4f}")
    torch.save(agent.state_dict(), f'./weights/{train_name}.pth')

    print(" ---------- Training Finished ----------")

    # Final evaluation
    print(" ---------- Final Evaluation ----------")
    final_performance = evaluate_performance(env, agent, num_episodes=100)
    print(f"Final average reward: {final_performance['avg_reward']:.2f}")
    print(f"Final average moving cost: {final_performance['moving_cost']:.4f}")
    print(f"Final average detour percentage: {final_performance['detour_percentage']:.2f}%")
    print(f"Final average computing time: {final_performance['computing_time']:.4f} s/step")
    print(f"Total failed paths: {final_performance['failed_paths']}")
    print(f"Total goals reached: {final_performance['agent_reached_goal']}")

    return agent
