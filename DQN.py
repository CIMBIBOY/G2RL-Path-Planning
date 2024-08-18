from cnn_for_dqn import CNNLSTMActor
import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from PER import PrioritizedReplayBuffer

class Agent:
    def __init__(self, enviroment, model, total_training_steps = 100000, device = 'cuda', batch_size = 32):
        
        # The number of states is the number of cells in the environment
        self._state_size = enviroment.n_states
        self._action_space = enviroment.f_action_space()
        self._action_size = enviroment.n_actions
        # Experience replay pool
        self.batch_size = batch_size
        self.expirience_replay = PrioritizedReplayBuffer(capacity=10000)
        
        # Initialize discount and exploration rate
        self.gamma = 0.6
        self.epsilon_initial = 1
        self.epsilon_final = 0.1
        # Set of thresholds where print a message is displayed
        self.epsilon_thresholds = {0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1}
        
        self.current_step = 0
        self.training_steps = total_training_steps

        self.device = torch.device(device)
        self.q_network = model.to(self.device)
        self.target_network = type(model)(30, 30, 4, 3).to(self.device)

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
        experience = (state, action, reward, next_state, terminated)
        self.expirience_replay.add(experience)

    def _build_compile_model(self):
        # Build initialization network model
        model = CNNLSTMActor(30,30,4,3)
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

        for threshold in sorted(self.epsilon_thresholds, reverse=True):
            if self.epsilon == threshold:
                print(f"Exploration value reached {threshold} e-greedy value")
                self.epsilon_thresholds.remove(threshold)
                break

        # take action
        if np.random.rand() <= self.epsilon or state.shape != (1,4,30,30,4):
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
        states, actions, rewards, next_states, dones, indices, weights = self.expirience_replay.sample(self.batch_size)

        states = torch.from_numpy(np.array(states)).float().to(self.device)
        next_states = torch.from_numpy(np.array(next_states)).float().to(self.device)
        actions = torch.from_numpy(np.array(actions)).long().to(self.device)
        rewards = torch.from_numpy(np.array(rewards)).float().to(self.device)
        dones = torch.from_numpy(np.array(dones)).bool().to(self.device)
        weights = torch.from_numpy(np.array(weights)).float().to(self.device)

        states = states.squeeze(1)  # Remove the unnecessary dimension
        next_states = next_states.squeeze(1)

        self.optimizer.zero_grad()

        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))

        with torch.no_grad():
            next_q_values = self.target_network(next_states)
            max_next_q_values = torch.max(next_q_values, dim=1)[0]
            # Bellman Equation
            target_q_values = rewards + self.gamma * max_next_q_values * (~dones)

        loss = (weights * self.criterion(current_q_values, target_q_values.unsqueeze(1))).mean()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)

        self.optimizer.step()
        self.scheduler.step()

        # Update priorities
        errors = torch.abs(target_q_values.unsqueeze(1) - current_q_values).cpu().detach().numpy()
        self.expirience_replay.update_priorities(indices, errors)

        return loss.item()