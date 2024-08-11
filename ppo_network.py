# ppo_training.py

import torch
import numpy as np
from cnn import CNNLSTMModel
from maskPPO import MaskPPOAgent
from eval import evaluate_performance
import progressbar
import time
from model_summary import print_model_summary

# PPO training script
def ppo_training(env, num_episodes=1144, timesteps_per_episode=1000, save_images=False, device='cpu', model_weights_path=None, batch_size=32, train_name='train', cmd_log=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Currently running training on device: {device}")

    # Initialize the PPO agent with its network
    agent = MaskPPOAgent(env, CNNLSTMModel(30, 30, 4, 3).to(device), device=device, batch_size=batch_size)
    
    # Load model weights if provided
    if model_weights_path:
        try:
            agent.model.load_state_dict(torch.load(model_weights_path, map_location=device, weights_only=True))
            print(f"Loaded model weights from: {model_weights_path}")
            time.sleep(2)
        except Exception as e:
            print(f"Error loading model weights: {e}")
            time.sleep(2)
    
    # Print model summary
    print_model_summary(agent.model, (batch_size, 4, 30, 30, 4), batch_size)

    all_episode_rewards = []
    bar_steps = 0
    cmd_print = cmd_log
    start_time = time.time()
    bar_bool = False  # Set to True if you want progress bars
    total_timesteps = num_episodes * timesteps_per_episode
    batch_rewards = []
    batch_start_time = time.time()
    steps = 0
    render = 0

    try:
        for e in range(num_episodes):
            _, state = env.reset()
            # Render the environment if enabled
            if env.pygame_render:
                env.render()  
            episode_reward = 0
            done = False
            if e == 0: 
                print(" ---------- Training Started ----------")

            if (e == 0 or e + 1 % cmd_print == 0) and bar_bool:
                bar = progressbar.ProgressBar(maxval=cmd_print * timesteps_per_episode, 
                                    widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
                bar.start()

            for timestep in range(timesteps_per_episode):
                # Adjust learning rate
                step = e * timesteps_per_episode + timestep
                agent.adjust_learning_rate(step, total_timesteps)

                # Select action
                action, log_prob, value = agent.select_action(state)
                env.last_action = action # store the last action for better maskin[[g

                # Step environment
                next_state, agent_pos, reward, done = env.step(action)

                # Render the environment if enabled
                if env.pygame_render:
                    env.render()  
                if save_images:
                    env.render_video(train_name, render)
                render += 1

                if state.shape != (1, 1, 30, 30, 4): 
                    agent.store(state, action, reward, next_state, done, log_prob, value)
                episode_reward += reward
                batch_rewards.append(reward)

                if done:
                    break

                state = next_state

                if (e + 1) % cmd_print == 0 and bar_bool:
                    bar_steps += 1
                    bar.update(bar_steps)

                steps += 1

            # Update agent using experiences from the replay buffer
            agent.replay_buffer_update()

            all_episode_rewards.append(episode_reward)

            if (e + 1) % cmd_print == 0:
                end_time = time.time()
                computing_time = (end_time - start_time)
                steps = 0
                if bar_bool: bar.finish()
                print(f" -------------------- Episode: {e + 1} -------------------- \nReward: {episode_reward:.2f}, Computing time: {computing_time:.4f} s/{cmd_log} epochs")
                start_time = time.time()
                # Log ppo data
                logs = agent.get_logs()
                print(f"Policy Loss: {logs['policy_loss']:.4f}")
                print(f"Value Loss: {logs['value_loss']:.4f}")
                print(f"Entropy: {logs['entropy']:.4f}")
                print(f"KL Divergence: {logs['approx_kl_div']:.4f}\n")

            if (e + 1) % 50 == 0:
                batch_end_time = time.time()
                batch_computing_time = (batch_end_time - batch_start_time) / 60
                # Save the model weights every 100 episodes
                print(f"\n---------------------------------------- {e+1}'th episode ----------------------------------------\n")
                print(f"Reward: {np.mean(batch_rewards):.2f}, Computing time: {batch_computing_time:.2f} min/100 epochs\nGoal reached for start-goal pair: {env.arrived} times, Number of collisions: {env.collisions}\n")
                print(f"Terminations casued by - Reached goals: {env.terminations[0]}, No guidance information: {env.terminations[1]}, Max steps reached: {env.terminations[2]}")
                torch.save(agent.model.state_dict(), f'./weights/ppo_model_{device}_{train_name}.pth')
                # Log ppo data
                batch_rewards = []
                logs = agent.get_logs()
                print(f"100 epoch Policy Loss: {logs['policy_loss']:.4f}")
                print(f"100 epoch Value Loss: {logs['value_loss']:.4f}")
                print(f"100 epoch Entropy: {logs['entropy']:.4f}")
                print(f"100 epoch KL Divergence: {logs['approx_kl_div']:.4f}\n")

            # Re-randomize the start and goal cells of all dynamic obstacles after 50 episodes
            # e % 10000 == 0:
            #   
            # TODO: implement loading of a different map for better generalization

        print(" ---------- Training Finished ----------")

        # Evaluate the PPO agent
        print(" ---------- Evaluating Performance ----------")
        performance_metrics = evaluate_performance(env, agent, num_episodes=154)
        print(performance_metrics)

    finally:
        # Save the training metrics
        np.save(f'./models/ppo_episode_rewards_{device}.npy', all_episode_rewards)

        # Log final results
        logs = agent.get_logs()
        print(f"Final Policy Loss: {logs['policy_loss']:.4f}")
        print(f"Final Value Loss: {logs['value_loss']:.4f}")
        print(f"Final Entropy: {logs['entropy']:.4f}")
        print(f"Final KL Divergence: {logs['approx_kl_div']:.4f}")
