import numpy as np
import random
from collections import deque
from network import MLP, CNN
import torch
import copy

class nn_Agent:
    def __init__(self, enviroment, model, device):
        self.enviroment = enviroment
        
        # 状态数是环境的格子数
        self._state_size = enviroment.n_states
        self._action_space = enviroment.action_space()
        self._action_size = enviroment.n_actions
        # 经验回放池
        self.expirience_replay = deque(maxlen=5)
        self.reward_replay = deque(maxlen=5)
        
        # Initialize discount and exploration rate
        self.gamma = 0.6
        self.epsilon = 0.1
        
        # Build networks
        self.q_network = model
        self.target_network = model
        self.device = device
        self.criterion = torch.nn.MSELoss()
        self.optim = torch.optim.Adam(self.q_network.parameters(), lr=3e-5)

    def store(self, state, action, reward, next_state, terminated):
        # 放入经验回放池
        self.expirience_replay.append((state, action, reward, next_state, terminated))
        if reward > 0:
            self.reward_replay.append((state, action, reward, next_state, terminated))
    
    def _build_compile_model(self):
        # 构建初始化网络模型
        model = MLP(self._state_size, self._action_size)
        return model

    def alighn_target_model(self):
        # 更新目标网络
        self.target_network.load_state_dict(self.q_network.state_dict(), strict=True)
    
    def act(self, state):
        # 采取动作
        if np.random.rand() <= self.epsilon:
            return random.choice(self._action_space)
        
        q_values = self.q_network(state)
        # import pdb; pdb.set_trace()
        return torch.argmax(q_values).item()

    def retrain(self, batch_size):
        # 利用经验池训练
        minibatch = random.sample(self.expirience_replay, len(self.expirience_replay))
        
        for state, action, reward, next_state, terminated in minibatch:
            input = torch.from_numpy(state.T).float().to(self.device)
            y_pred = self.q_network(input)
            labels = y_pred.clone()
            if terminated:
                labels[action] = reward
            else:
                next_input = torch.from_numpy(next_state.T).float().to(self.device)
                t = self.target_network(next_input)
                labels[action] = reward + self.gamma * torch.max(t).item()
            
            # 梯度下降
            loss = self.criterion(y_pred, labels)
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
        # 利用奖励池训练
        minibatch = random.sample(self.reward_replay, min(batch_size, len(self.reward_replay)))

        for state, action, reward, next_state, terminated in minibatch:
            input = torch.from_numpy(state.T).float().to(self.device)
            y_pred = self.q_network(input)
            labels = y_pred.clone()
            if terminated:
                labels[action] = reward
            else:
                next_input = torch.from_numpy(next_state.T).float().to(self.device)
                t = self.target_network(next_input)
                labels[action] = reward + self.gamma * torch.max(t).item()
            
            # 梯度下降
            loss = self.criterion(y_pred, labels)
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
        