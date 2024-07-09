from cnn import CNNLSTMModel
import numpy as np
import random
from collections import deque
from WarehouseEnv import WarehouseEnvironment
from model_summary import print_model_summary
import torch 

class Agent:
    def __init__(self, enviroment, model):
        
        # The number of states is the number of cells in the environment
        self._state_size = enviroment.n_states
        self._action_space = enviroment.action_space()
        self._action_size = enviroment.n_actions
        # Experience replay pool
        self.expirience_replay = deque(maxlen=4)
        
        # Initialize discount and exploration rate
        self.gamma = 0.6
        self.epsilon = 0.1
        
        # Build networks
        self.q_network = model
        self.target_network = model
        # self.alighn_target_model()

    def store(self, state, action, reward, next_state, terminated):
        # Store in experience replay pool
        self.expirience_replay.append((state, action, reward, next_state, terminated))
    
    def _build_compile_model(self):
        # Build initialization network model
        model = CNNLSTMModel(48,48,4,4)
        return model

    def alighn_target_model(self):
        # Update target network
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def act(self, state):
        state_tensor = torch.from_numpy(state)

        # take action
        if np.random.rand() <= self.epsilon:
            action = random.choice(self._action_space)
            # print(f"random action: {action}")
            return action
        
        q_values = self.q_network.forward(state_tensor)
        m_action = np.argmax(q_values[0].detach().numpy())
        # print(f"model action: {m_action}")
        return m_action

    def retrain(self, batch_size):
        # Leverage experience pool training
        minibatch = random.sample(self.expirience_replay, batch_size)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.RMSprop(self.q_network.parameters(), lr=3e-5)
        
        for state, action, reward, next_state, terminated in minibatch:
            
            state_tensor = torch.from_numpy(state)
            next_state_tensor = torch.from_numpy(next_state)
            target = self.q_network(state_tensor)
            
            target_val = target.clone().detach()
            
            with torch.no_grad():
                if terminated:
                    target_val[0][action] = reward
                else:
                    t = self.target_network(next_state_tensor)
                    target_val[0][action] = reward + self.gamma * torch.max(t).item()
            
            optimizer.zero_grad()
            loss = criterion(self.q_network(state_tensor), target_val)
            loss.backward()
            optimizer.step()
