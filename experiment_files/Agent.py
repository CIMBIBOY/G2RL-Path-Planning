import numpy as np
import random
from collections import deque
from network import MLP, CNN
import torch
import copy

class nn_Agent:
    def __init__(self, enviroment, model, device):
        self.enviroment = enviroment
        
        # Number of states is the number of cells in the environment
        self._state_size = enviroment.n_states
        self._action_space = enviroment.action_space()
        self._action_size = enviroment.n_actions
        # Experience replay pool
        self.expirience_replay = deque(maxlen=100)
        self.reward_replay = deque(maxlen=5)
        
        # Initialize discount and exploration rate
        self.gamma = 0.6
        self.epsilon = 0.1
        
        # Build networks
        self.q_network = model
        self.target_network = model
        self.device = device
        self.criterion = torch.nn.MSELoss()
        self.optim = torch.optim.RMSprop(self.q_network.parameters(), lr=3e-5)

    def store(self, state, action, reward, next_state, terminated):
        # Store in experience replay pool
        self.expirience_replay.append((state, action, reward, next_state, terminated))
        if reward > 0:
            self.reward_replay.append((state, action, reward, next_state, terminated))
    
    def _build_compile_model(self):
        # Initialization network 
        model = MLP(self._state_size, self._action_size)
        return model

    def load_target_model(self):
        # Update target network
        self.target_network.load_state_dict(self.q_network.state_dict(), strict=True)
    
    def act(self, state):
        # Random action choice 
        if np.random.rand() <= self.epsilon:
            return random.choice(self._action_space)
        
        q_values = self.q_network(state)
        # import pdb; pdb.set_trace()
        return torch.argmax(q_values).item()

    def retrain(self, batch_size):
        # Leverage experience pool training
        minibatch = random.sample(self.expirience_replay, batch_size)
        loss = 0
        
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
            
            # gradient descent 
            loss += self.criterion(y_pred, labels)
        self.optim.zero_grad()
        loss.backward() # retain_graph = true? - lehet belehív a c++ back engine-be
        self.optim.step()
        return loss.item()/len(minibatch)
    
        # Training with a reward pool

        # minibatch = random.sample(self.reward_replay, min(batch_size, len(self.reward_replay)))

        # for state, action, reward, next_state, terminated in minibatch:
        #     input = torch.from_numpy(state.T).float().to(self.device)
        #     y_pred = self.q_network(input)
        #     labels = y_pred.clone()
        #     if terminated:
        #         labels[action] = reward
        #     else:
        #         next_input = torch.from_numpy(next_state.T).float().to(self.device)
        #         t = self.target_network(next_input)
        #         labels[action] = reward + self.gamma * torch.max(t).item()

        #     loss = self.criterion(y_pred, labels)
        #     self.optim.zero_grad()
        #     loss.backward()
        #     self.optim.step()
        