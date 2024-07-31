import argparse
import numpy as np
import progressbar
import pickle
import random
import time
import sys

from WarehouseEnv import WarehouseEnvironment
from DQN import Agent
from cnn import CNNLSTMModel
from model_summary import print_model_summary
from eval import evaluate_performance

# python3 main.py --render off --method dqn --steps 100000
# or
# python3 main.py --render off --method qnet --episode 100000

def dqn_training(env, num_episodes=1144, timesteps_per_episode = 200):
    agent = Agent(env, CNNLSTMModel(30,30,4,4))
    batch_size = 3
    image = 0
    N = 100

    print_model_summary(agent.q_network, (batch_size, 1, 30, 30, 4), batch_size)

    all_episode_rewards = []
    all_episode_losses = []
    batch_episode_time = 0
    try:
        for e in range(num_episodes):
            _, state = env.reset()
            episode_reward = 0
            batch_reward = 0
            episode_loss = 0
            batch_loss = 0
            terminated = False
            
            timesteps_per_episode =  5 * env.agent_path_len
            bar = progressbar.ProgressBar(maxval=timesteps_per_episode, 
                                  widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
            bar.start()
            start_time = time.time()
            steps = 1

            print(env.agent_path_len)
            
            for timestep in range(timesteps_per_episode):
                action = agent.act(state)
                next_state, _, reward, terminated = env.step(action) 
                agent.store(state, action, reward, next_state, terminated)

                # Render the environment
                if env.pygame_render == True:
                    env.render()
                env.render_video(5, image)
                image += 1
                
                if terminated:
                    # agent.align_target_model()
                    break
                
                state = next_state
                    
                if len(agent.expirience_replay) > batch_size:
                    loss = agent.retrain(batch_size)
                    episode_loss += loss
                    batch_loss += episode_loss
                    
                episode_reward += reward
                batch_reward += episode_reward
                
                if timestep % 1 == 0:
                    bar.update(timestep + 1)
                
                steps += 1
  
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
            if (e + 1) % 20 == 0: 
                batch_episode_time = batch_episode_time / 60
                print(f"\n---------- 20 episode periods ----------\n Reward: {batch_reward:.2f}, Loss: {batch_loss:.2f}, Computing time: {computing_time:.2f} min,  Goal reached: {env.arrived} times\n")
                batch_episode_time = batch_loss = batch_reward = 0
                
        print(" ---------- Training Finished ----------")

        # Eval of Deep Q-network
        print(" ---------- Evaluating Performance ----------")
        performance_metrics = evaluate_performance(env, agent, num_episodes=154)
        print(performance_metrics)

    finally:
        # Save the training metrics
        with open('./models/dqn_episode_rewards.pkl', 'wb') as f:
            pickle.dump(all_episode_rewards, f)
        with open('./models/dqn_episode_losses.pkl', 'wb') as f:
            pickle.dump(all_episode_losses, f)


def q_learning_training(env, num_episodes=100000):
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
                env.render_video(5,image)
                image = image + 1
                
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
                goal_reached_rate = sum(all_goal_reached[-100:])/100
                print(f"Episode: {i}, Reward: {reward:.2f}, Avg Loss: {avg_loss:.4f}, Goal Reached Rate: {goal_reached_rate:.2f}, Computing time: {computing_time:.4f} s/step")
            if i % 20 == 0: 
                batch_episode_time = batch_episode_time / 60
                print(f"Minutes of computing 20 episodes: {computing_time:.2f} min")
                batch_episode_time = 0


        print("Training finished.\n")

        # Eval of simple Q-network
        print(" ---------- Evaluating Performance ----------")
        performance_metrics = evaluate_performance(env, q_table, num_episodes=100)
        print(performance_metrics)

    finally: 
        with open('./models/q_learning_table.pkl','wb') as f:
                pickle.dump(q_table, f)

        with open('./models/rewards_window.pkl','wb') as f:
            pickle.dump(rewards_window, f)

        with open('./models/all_rewards.pkl','wb') as f:
            pickle.dump(all_rewards, f)

        with open('./models/all_losses.pkl','wb') as f:
            pickle.dump(all_losses, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Choose between CCNLSTM agent supported DQN model or traditional Q-Learning')
    parser.add_argument('--method', type=str, choices=['dqn', 'qnet'], default='dqn',
                        help='Choose the training method: deep Q-network or traditional Q-network')
    parser.add_argument('--render', type=str, choices=['on', 'off'], default='off',
                        help='Choose to visualize the training in pygame? Options: --render on, or --render off')
   # Add an argument for the number of episodes
    parser.add_argument('--steps', type=int, default=1000,
                        help='Number of episodes for training.')
    args = parser.parse_args()

    num_ep = args.steps

    if args.render == 'on':
        env = WarehouseEnvironment(pygame_render=True)
    elif args.render == 'off':
        env = WarehouseEnvironment(pygame_render=False)
    else:
        print("Render automatically set to False!")
        env = WarehouseEnvironment(pygame_render=False)

    if args.method == 'dqn':
        dqn_training(env, num_episodes = num_ep)
    elif args.method == 'qnet':
        q_learning_training(env, num_episodes = num_ep)   
    else: print("No method choosen or type error in parsing argument! Please eaither use command: \npython main.py --method dqn \nor\n python main.py --method qnet")

    env.close()

    