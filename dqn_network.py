import torch
from model_summary import print_model_summary
from DQN import Agent
from cnn import CNNLSTMModel
import progressbar
import time
from eval import evaluate_performance
import numpy as np

def dqn_training(env, num_episodes=1144, timesteps_per_episode = 33, save_images = False, metal = 'cpu', model_weights_path=None, batch_size = 32):
    
    # Load model weights if provided
    if model_weights_path:
        agent.q_network.load_state_dict(torch.load(model_weights_path))
        print(f"Loaded model weights from: {model_weights_path}")
    
    agent = Agent(env, CNNLSTMModel(30,30,4,4), metal = metal)
    image = 0
    N = 100

    print_model_summary(agent.q_network, (batch_size, 1, 30, 30, 4), batch_size)

    all_episode_rewards = []
    all_episode_losses = []
    batch_episode_time = 0
    batch_loss = 0
    batch_reward = 0

    try:        
        for e in range(num_episodes):

            if num_episodes < 2: print(state.device)
            _, state = env.reset()
            episode_reward = 0
            episode_loss = 0
            terminated = False
            
            timesteps_per_episode =  3 * env.agent_path_len
            bar = progressbar.ProgressBar(maxval=timesteps_per_episode, 
                                widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
            bar.start()
            start_time = time.time()
            steps = 1

            if num_episodes < 2: 
                print(f"Currently running train on device: {state.device}")
            
            for timestep in range(timesteps_per_episode):

                action = agent.act(state)
                next_state, _, reward, terminated = env.step(action) 
                agent.store(state, action, reward, next_state, terminated)

                # Check for rendering toggle
                if env.pygame_render:
                    env.render()
                if save_images:
                    env.render_video(5, image)
                    image += 1
                
                if terminated:
                    # agent.align_target_model()
                    break
                
                state = next_state
                    
                if len(agent.expirience_replay) > batch_size:
                    loss = agent.retrain(batch_size)
                    episode_loss += loss
                    
                episode_reward += reward
                
                if timestep % 1 == 0:
                    bar.update(timestep + 1)
                
                steps += 1

            batch_loss += episode_loss
            batch_reward += episode_reward

            end_time = time.time()
            bar.finish()
            computing_time = (end_time - start_time) / steps
            batch_episode_time += computing_time
            
            # Log the episode-wise metrics
            all_episode_rewards.append(episode_reward)
            all_episode_losses.append(episode_loss)

            # Step the scheduler every N episodes
            if (e + 1) % N == 0:
                agent.scheduler.step()

            if (e + 1) % 1 == 0:
                print(f"Episode: {e + 1}, Reward: {episode_reward:.2f}, Loss: {episode_loss:.2f}, Computing time: {computing_time:.4f} s/step")

            if (e + 1) % 100 == 0: 
                batch_episode_time = batch_episode_time 
                print(f"\n---------- 100 episode periods ----------\n Reward: {batch_reward:.2f}, Loss: {batch_loss:.2f}, Computing time: {computing_time:.2f} sec/100 epochs,  Goal reached: {env.arrived} times\n")
                batch_episode_time = batch_loss = batch_reward = 0
                
                # Save the model weights
                torch.save(agent.q_network.state_dict(), f'./weights/dqn_model_{metal}.pth')  # For DQN
                
        print(" ---------- Training Finished ----------")

        # Eval of Deep Q-network
        print(" ---------- Evaluating Performance ----------")
        performance_metrics = evaluate_performance(env, agent, num_episodes=154)
        print(performance_metrics)

    finally:
        # Save the training metrics
        np.save(f'./models/dqn_episode_rewards_{metal}.npy', all_episode_rewards)
        np.savez(f'./models/dqn_metrics_{metal}.npz', rewards=all_episode_rewards, losses=all_episode_losses)
        