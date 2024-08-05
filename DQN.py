from cnn import CNNLSTMModel
import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

class Agent:
    def __init__(self, enviroment, model, total_training_steps = 200000, metal = 'cuda'):
        
        # The number of states is the number of cells in the environment
        self._state_size = enviroment.n_states
        self._action_space = enviroment.action_space()
        self._action_size = enviroment.n_actions
        # Experience replay pool
        self.batch_size = 32
        self.expirience_replay = deque(maxlen=10000)
        
        # Initialize discount and exploration rate
        self.gamma = 0.6
        self.epsilon_initial = 1
        self.epsilon_final = 0.1
        self.current_step = 0
        self.training_steps = total_training_steps

        self.device = torch.device(metal)
        self.q_network = model.to(self.device)
        self.target_network = type(model)(30, 30, 4, 4).to(self.device)

        self.target_network.load_state_dict(self.q_network.state_dict())  # Copy weights from q_network to target_network
        self.target_network.eval()  # Set target network to evaluation mode
        self.tau = 0.001  # Soft update parameter
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=1e-3)
        
        # Initialize learning rate scheduler
        self.scheduler = StepLR(self.optimizer, step_size=100, gamma=0.1)
        
        # Initialize loss function
        self.criterion = nn.MSELoss()

        # Track random and network actions
        self.rand_act = 0
        self.netw_act = 0

    def store(self, state, action, reward, next_state, terminated):
        # Store in experience replay pool
        self.expirience_replay.append((state, action, reward, next_state, terminated))
    
    def _build_compile_model(self):
        # Build initialization network model
        model = CNNLSTMModel(48,48,4,4)
        return model

    def align_target_model(self):
        # Soft update
        for target_param, param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data)
    
    def act(self, state):
        self.current_step += 1
        state_tensor = torch.from_numpy(state).float().to(self.device)

        # Update epsilon
        self.epsilon = self.epsilon_initial - (self.epsilon_initial - self.epsilon_final) * min(1, self.current_step / self.training_steps)
        epsilon_final = False
        if self.epsilon == 0.1 and epsilon_final == False:
            print("Exploration value reached 0.1 e-greedy value")
            epsilon_final = True

        # take action
        if np.random.rand() <= self.epsilon or self.current_step < 4:
            action = random.choice(self._action_space)
            self.rand_act += 1
            # print(f"random action: {action}")
            return action
        
        q_values = self.q_network.forward(state_tensor)
        # Move the tensor to CPU before converting to numpy
        m_action = np.argmax(q_values[0].cpu().detach().numpy())
        self.netw_act += 1
        # print(f"model action: {m_action}")
        return m_action

    def retrain(self):
        minibatch = random.sample(self.expirience_replay, self.batch_size)
        
        total_loss = 0.0
        
        self.optimizer.zero_grad()  # Move this outside the loop

        for state, action, reward, next_state, terminated in minibatch:
            state = torch.from_numpy(state).float().to(self.device)
            next_state = torch.from_numpy(next_state).float().to(self.device)
        
            with torch.no_grad():
                current_q_values = self.q_network(state).detach()
            
            with torch.no_grad():
                if not terminated:
                    next_q_values = self.target_network(next_state)
                    target_q_value = reward + self.gamma * torch.max(next_q_values).item()
                else:
                    target_q_value = reward
            
            target = current_q_values.clone()
            target[0][action] = target_q_value
            
            current_q_values = self.q_network(state)  # Recompute with grad
            loss = self.criterion(current_q_values, target)
            loss.backward()
            
            total_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        self.align_target_model()  # Move this outside the loop

        self.scheduler.step()
        
        return total_loss / self.batch_size