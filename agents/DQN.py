from agents.cnn_for_dqn import CNNLSTMActor
import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from helpers.PER import PrioritizedReplayBuffer

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
        self.target_network = type(model)(7).to(self.device)

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

        # Initialize LSTM states
        self.lstm_state = self.init_lstm_states()

    def store(self, state, action, reward, next_state, terminated):
        # Store in experience replay pool
        experience = (state, action, reward, next_state, terminated)
        self.expirience_replay.add(experience)

    def _build_compile_model(self):
        # Build initialization network model
        model = CNNLSTMActor(7)
        return model

    def align_target_model(self):
        # Soft update
        for target_param, param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data)
    
    def act(self, state, termination_flag):
        self.current_step += 1
        state_tensor = torch.from_numpy(state).float().to(self.device).unsqueeze(0)  # Add batch dimension

        # Update epsilon
        self.epsilon = self.epsilon_initial - (self.epsilon_initial - self.epsilon_final) * min(1, self.current_step / self.training_steps)

        for threshold in sorted(self.epsilon_thresholds, reverse=True):
            if self.epsilon == threshold:
                print(f"Exploration value reached {threshold} e-greedy value")
                self.epsilon_thresholds.remove(threshold)
                break

        # Take action
        if np.random.rand() <= self.epsilon:
            action = random.choice(self._action_space)
            self.rand_act += 1
            return action
            
        next_done = torch.Tensor(termination_flag).to(self.device) # Assuming no episode is done during this single step
        # Before the forward pass
        self.lstm_state = self.detach_lstm_state(self.lstm_state)
        q_values, self.lstm_state = self.q_network(state_tensor, self.lstm_state, next_done)  # Update lstm_state
        m_action = np.argmax(q_values[0].cpu().detach().numpy())
        self.netw_act += 1
        return m_action

    def retrain(self):
        torch.autograd.set_detect_anomaly(True)
        states, actions, rewards, next_states, dones, indices, weights = self.expirience_replay.sample(self.batch_size)

        states = torch.from_numpy(np.array(states)).float().to(self.device)
        next_states = torch.from_numpy(np.array(next_states)).float().to(self.device)
        actions = torch.from_numpy(np.array(actions)).long().to(self.device)
        rewards = torch.from_numpy(np.array(rewards)).float().to(self.device)
        dones = torch.from_numpy(np.array(dones)).float().to(self.device)
        weights = torch.from_numpy(np.array(weights)).float().to(self.device)

        states = states.squeeze(1)
        next_states = next_states.squeeze(1)

        self.optimizer.zero_grad()

        # Before the forward pass
        self.lstm_state = self.detach_lstm_state(self.lstm_state)
        # Forward pass with Q-network
        current_q_values, self.lstm_state = self.q_network(states, self.lstm_state, dones)  # Update lstm_state
        # Forward pass with Q-network
        current_q_values = current_q_values.gather(1, actions.unsqueeze(1))

        with torch.no_grad():
            # Forward pass with target network
            # Before the forward pass
            self.lstm_state = self.detach_lstm_state(self.lstm_state)
            next_q_values, _ = self.target_network(next_states, self.lstm_state, dones)  # No need to update lstm_state in the target network
            max_next_q_values = torch.max(next_q_values, dim=1)[0]
            # Bellman Equation
            target_q_values = rewards + self.gamma * max_next_q_values * (~dones.bool())

        loss = (weights * self.criterion(current_q_values, target_q_values.unsqueeze(1))).mean()
        loss.backward(retain_graph=True)

        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()

        # Update priorities
        errors = torch.abs(target_q_values.unsqueeze(1) - current_q_values).cpu().detach().numpy()
        self.expirience_replay.update_priorities(indices, errors)

        return loss.item()
    
    def init_lstm_states(self):
        return (
            torch.zeros(self.q_network.lstm.num_layers, self.batch_size, self.q_network.lstm.hidden_size).to(self.device),
            torch.zeros(self.q_network.lstm.num_layers, self.batch_size, self.q_network.lstm.hidden_size).to(self.device)
        )
    
    def detach_lstm_state(self, lstm_state):
        return (lstm_state[0].detach(), lstm_state[1].detach())

