from PIL import Image
import numpy as np
from heapq import heappop, heappush
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange

from network import MLP, CNN
from Env import StaticEnvironment
from Agent import nn_Agent


if __name__ == '__main__':
    # GPU设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_path = "models/model_test.pkl"
    load_path = "models/1219_1.pkl"
    # 环境和智能体初始化
    env = StaticEnvironment()
    env.is_render = True
    state = env.reset()
    model = CNN(env.dim_states, env.n_actions).to(device)
    model.load_state_dict(torch.load(load_path))
    agent = nn_Agent(env, model, device)
    # 测试
    input = torch.from_numpy(state.T).float().to(device)
    print(input.size())
    # 训练参数设置
    batch_size = 4
    num_of_episodes = 100
    timesteps_per_episode = 1000
    
    num_finished_episodes = 0
    # 训练
    for i in trange(num_of_episodes):
        # Reset the enviroment
        state = env.reset()
        for timestep in range(timesteps_per_episode):
            # Run Action
            input = torch.from_numpy(state.T).float().to(device)
            # import pdb; pdb.set_trace()
            action = agent.act(input)
            # Take action    
            reward, next_state, done = env.step(action)
            # Store the transition
            agent.store(state, action, reward, next_state, done)
            # Retrain the agent
            agent.retrain(batch_size)
            # Update the state
            state = next_state
            # Update the plot
            if done:
                break
        # Update the target network with new weights
        # print(timestep)
        agent.alighn_target_model()
        if timestep < timesteps_per_episode-1:
            num_finished_episodes += 1
    print("num_finished_episodes:", num_finished_episodes, "percentage:", num_finished_episodes/num_of_episodes)
    torch.save(agent.q_network.state_dict(), save_path)
