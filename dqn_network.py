import torch
from model_summary import print_model_summary
from DQN import Agent
from cnn import CNNLSTMModel
import progressbar
import time
from eval import evaluate_performance
import numpy as np
from train_utils import debug_start, debug_end

# DQN training script
def dqn_training(env, num_episodes=1144, timesteps_per_episode = 33, save_images = False, metal = 'cpu', model_weights_path=None, batch_size = 32, train_name = 'train', cmd_log = 5, explore = 200000):
    # Set the device to CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Currently running training on device: {device}")

    # Initialize the agent with its network
    agent = Agent(env, CNNLSTMModel(30,30,4,3).to(device), total_training_steps=explore, metal = device)
    agent.batch_size = batch_size
    N = 50

   # Load model weights if provided
    if model_weights_path:
        try:
            agent.q_network.load_state_dict(torch.load(model_weights_path, map_location=device, weights_only=True))
            print(f"Loaded model weights from: {model_weights_path}")
            time.sleep(2)
        except Exception as e:
            print(f"Error loading model weights: {e}")
            time.sleep(2)
    
    print_model_summary(agent.q_network, (batch_size, 4, 30, 30, 4), batch_size)

    all_episode_rewards = []
    all_episode_losses = []
    bar_steps = 0
    cmd_print = cmd_log
    batch_loss = []
    batch_rewards = []
    batch_start_time = time.time()
    start_time = time.time()
    bar_bool = False
    render = 0
    steps = 0

    try:        
        for e in range(num_episodes):
            _, state = env.reset()
            # Render the environment if it's enabled
            if env.pygame_render:
                env.render()
            episode_reward = 0
            episode_loss = 0
            terminated = False

            if e == 0: 
                print(" ---------- Training Started ----------")

            timesteps_per_episode =  4 * env.agent_path_len

            if (e == 0 or e + 1 % cmd_print == 0) and bar_bool:
                bar = progressbar.ProgressBar(maxval=cmd_print * timesteps_per_episode, 
                                    widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
                bar.start()
            
            for timestep in range(timesteps_per_episode):
                # Compute action
                action = agent.act(state)
                # Step the environment
                next_state, _, reward, terminated = env.step(action)

                # Render the environment if it's enabled
                if env.pygame_render:
                    env.render()
                if save_images:
                    env.render_video(train_name, render)
                render += 1

                # Store if shape is valid for the CNN input
                if state.shape != (1, 1, 30, 30, 4): 
                    agent.store(state, action, reward, next_state, terminated)
                
                if terminated:
                    agent.align_target_model()
                    break
                
                state = next_state
                    
                # stime = debug_start(timestep, 'retrain')
                if len(agent.expirience_replay) > agent.batch_size:
                    loss = agent.retrain()
                    episode_loss += loss
                    batch_loss.append(loss)
                # debug_end(stime)
                    
                episode_reward += reward
                batch_rewards.append(reward)

                if bar_bool:
                    # Update progress bar
                    bar_steps += 1
                    if e % cmd_print < cmd_print - 1 or timestep == timesteps_per_episode - 1:
                        bar.update(bar_steps)

                steps += 1 

            # Log the episode-wise metrics
            all_episode_rewards.append(episode_reward)
            all_episode_losses.append(episode_loss)

            if (e + 1) % cmd_print == 0 and (e + 1) % N != 0:
                end_time = time.time()
                computing_time = (end_time - start_time) 
                if bar_bool: bar.finish()
                print(f" Episode: {e + 1}, Reward: {episode_reward:.2f}, Loss: {episode_loss:.4f}, Computing time: {computing_time:.4f} s/{cmd_log} epochs")
                # Reset and restart progress bar
                bar_steps = 0
                start_time = time.time()

            if (e + 1) % N == 0: 
                batch_end_time = time.time()
                batch_computing_time = (batch_end_time - batch_start_time) / 60
                start_end = (e+1)/50
                print(f"\n----------------------------------- {e+1}'th episode - {start_end}'th start-end pair -----------------------------------\n")
                print(f"Reward: {np.mean(batch_rewards):.2f},  Loss: {np.mean(batch_loss):.4f}, Computing time: {batch_computing_time:.2f} min/{N} epochs\nGoal reached for start-goal pair: {env.arrived} times, Steps taken in {N} epochs: {steps}, Random actions: {agent.rand_act}, Network actions: {agent.netw_act}\n")
                batch_start_time = time.time()
                print(f"Terminations casued by - Reached goals: {env.terminations[0]:.0f}, No guidance information: {env.terminations[1]:.0f}, Max steps reached: {env.terminations[2]:.0f}, Collisions with obstacles: {env.terminations[3]:.0f}\n")
                # Log ppo data
                steps = agent.rand_act = agent.netw_act = 0
                batch_rewards = []
                batch_loss = []
            
            if (e + 1) % 1000 == 0: 
                print(f"Is CUDA being used? {next(agent.q_network.parameters()).is_cuda}\n")
                # Save the model weights
                torch.save(agent.q_network.state_dict(), f'./weights/dqn_model_{metal}_{train_name}.pth')  # For DQN
                
        print(" ---------- Training Finished ----------")

        # Eval of Deep Q-network
        print(" ---------- Evaluating Performance ----------")
        performance_metrics = evaluate_performance(env, agent, num_episodes=200)
        print(performance_metrics)

    finally:
        # Save the training metrics
        np.save(f'./models/dqn_episode_rewards_{metal}.npy', all_episode_rewards)
        np.savez(f'./models/dqn_metrics_{metal}.npz', rewards=all_episode_rewards, losses=all_episode_losses)
        