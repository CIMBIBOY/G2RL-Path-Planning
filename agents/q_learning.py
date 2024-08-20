import torch
import random
import time
from eval.eval import evaluate_performance
import numpy as np


def q_learning_training(env, num_episodes=100000, save_images = False):
    q_table = np.zeros([env.n_states, env.n_actions])
    alpha, gamma, epsilon = 0.3, 0.9, 0.1
    rewards_window, all_rewards, all_losses = [], [], []
    image = 0
    batch_episode_time = 0

    try: 
        for i in range(1, num_episodes + 1):
            state, _ = env.reset()
            penalties, reward, episode_loss = 0, 0, 0
            done = False
            
            start_time = time.time()
            steps = 1

            while not done:
                action = random.choice(env.action_space()) if random.uniform(0, 1) < epsilon else np.argmax(q_table[state])
                _, next_state, reward, done = env.step(action)

                # pygame visulaization, image+video rendering and gif
                if env.pygame_render == True:
                    env.render()
                if save_images == True:    
                    env.render_video(5, image)
                    image += 1
                
                old_value = q_table[state, action]
                next_max = np.max(q_table[next_state])
                new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
                q_table[state, action] = new_value

                # Calculate loss as the absolute difference between old and new Q-values
                loss = abs(old_value - new_value)
                episode_loss += loss

                if reward <= -0.1:
                    penalties += 1
                
                state = next_state
                steps += 1
                
            end_time = time.time()
            computing_time = (end_time - start_time) / steps
            batch_episode_time += computing_time

            all_rewards.append(reward)
            all_losses.append(episode_loss)

            if i % 10 == 0:
                rewards_window.append(sum(all_rewards[-100:])/100)
                avg_loss = sum(all_losses[-100:])/100
                print(f"Episode: {i}, Reward: {reward:.2f}, Avg Loss: {avg_loss:.4f} Computing time: {computing_time:.4f} s/step")
            if i % 20 == 0: 
                batch_episode_time = batch_episode_time / 60
                print(f"Minutes of computing 20 episodes: {computing_time:.2f} min")
                batch_episode_time = 0

                torch.save(q_table, f'./models/q_table_episode_{i}.pkl')


        print("Training finished.\n")

        # Eval of simple Q-network
        print(" ---------- Evaluating Performance ----------")
        performance_metrics = evaluate_performance(env, q_table, num_episodes=100)
        print(performance_metrics)

    finally: 
        # Save the training metrics
        np.save('./models/q_learning_table.npy', q_table)
        np.save('./models/rewards_window.npy', rewards_window)
        np.save('./models/all_rewards.npy', all_rewards)
        np.save('./models/all_losses.npy', all_losses)