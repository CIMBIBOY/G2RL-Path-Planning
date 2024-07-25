import argparse
import numpy as np
import progressbar
import pickle
import random

from WarehouseEnv import WarehouseEnvironment
from DQN import Agent
from cnn import CNNLSTMModel
from model_summary import print_model_summary
from eval import evaluate_performance

# python main.py --method dqn
# or
# python main.py --method qnet

def dqn_training(env, num_episodes=1144, timesteps_per_episode=1000):
    agent = Agent(env, CNNLSTMModel(30,30,4,4))
    batch_size = 3
    image = 0

    print_model_summary(agent.q_network, (batch_size, 1, 30, 30, 4), batch_size)

    all_episode_rewards = []
    all_episode_losses = []

    for e in range(num_episodes):
        _, state = env.reset()
        episode_reward = 0
        episode_loss = 0
        terminated = False
        
        bar = progressbar.ProgressBar(maxval=timesteps_per_episode/10, 
                                      widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        
        for timestep in range(timesteps_per_episode):
            action = agent.act(state)
            next_state, _, reward, terminated = env.step(action) 
            agent.store(state, action, reward, next_state, terminated)
            state = next_state

            # Render the environment
            if env.pygame_render == True:
                env.render()
            env.render_video(5, image)
            image += 1
            # env.render_gif()
            
            if terminated:
                agent.alighn_target_model()
                break
                
            if len(agent.expirience_replay) > batch_size:
                loss = agent.retrain(batch_size)
                episode_loss += loss
                
            episode_reward += reward
            
            if timestep % 10 == 0:
                bar.update(timestep/10 + 1)
        
        bar.finish()
        
        # Log the episode-wise metrics
        all_episode_rewards.append(episode_reward)
        all_episode_losses.append(episode_loss)
        
        if (e + 1) % 1 == 0:
            print(f"Episode: {e + 1}, Reward: {episode_reward:.2f}, Loss: {episode_loss:.2f}")
    
    print(" ---------- Training Finished ----------")

    # Eval of Deep Q-network
    print(" ---------- Evaluating Performance ----------")
    performance_metrics = evaluate_performance(env, agent, num_episodes=154)
    print(performance_metrics)

    # Save the training metrics
    with open('./models/dqn_episode_rewards.pkl', 'wb') as f:
        pickle.dump(all_episode_rewards, f)
    with open('./models/dqn_episode_losses.pkl', 'wb') as f:
        pickle.dump(all_episode_losses, f)

def q_learning_training(env, num_episodes=100000):
    q_table = np.zeros([env.n_states, env.n_actions])
    alpha, gamma, epsilon = 0.3, 0.9, 0.1
    rewards_window, all_rewards = [], []
    image = 0

    for i in range(1, num_episodes + 1):
        state, _ = env.reset()
        epochs, penalties, reward = 0, 0, 0
        done = False
        
        while not done:
            action = random.choice(env.action_space()) if random.uniform(0, 1) < epsilon else np.argmax(q_table[state])
            _, next_state, reward, done = env.step(action)

            # pygame visulaization, image+video rendering and gif
            # Render the environment
            if env.pygame_render == True:
                env.render()
            env.render_video(5,image)
            image = image + 1
            # env.render_gif()
            
            old_value = q_table[state, action]
            next_max = np.max(q_table[next_state])
            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            q_table[state, action] = new_value

            if reward <= -0.1:
                penalties += 1
            
            state = next_state
            epochs += 1

        all_rewards.append(reward)

        if i % 100 == 0:
            rewards_window.append(sum(all_rewards[-100:])/100)
            print(f"Episode: {i} reward: {reward}")

    print("Training finished.\n")

    # Eval of simple Q-network
    print(" ---------- Evaluating Performance ----------")
    performance_metrics = evaluate_performance(env, q_table, num_episodes=100)
    print(performance_metrics)

    return q_table, rewards_window, all_rewards


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Choose between CCNLSTM agent supported DQN model or traditional Q-Learning')
    parser.add_argument('--method', type=str, choices=['dqn', 'qnet'], default='dqn',
                        help='Choose the training method: deep Q-network or traditional Q-network')
    args = parser.parse_args()

    env = WarehouseEnvironment()

    if args.method == 'dqn':
        dqn_training(env)
    elif args.method == 'qnet':
        q_table, rewards_window, all_rewards = q_learning_training(env)
        
        with open('./models/q_learning_table.pkl','wb') as f:
            pickle.dump(q_table, f)

        with open('./models/rewards_window.pkl','wb') as f:
            pickle.dump(rewards_window, f)

        with open('./models/all_rewards.pkl','wb') as f:
            pickle.dump(all_rewards, f)
    else: print("No method choosen or type error in parsing argument! Please eaither use command: python main.py --method dqn or python main.py --method qnet")

    env.close()

    