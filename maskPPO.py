import torch
import torch.nn as nn
import torch.optim as optim
from PER import PrioritizedReplayBuffer
import numpy as np
from train_utils import debug_start, debug_end

class MaskPPOAgent:
    def __init__(self, env, model, device='cpu', batch_size=32, mini_batch_size=8, epochs=4, gamma=0.99, gae_lambda=0.95, clip_epsilon=0.2, c1=0.5, c2=0.01, lr=3e-4):
        self.env = env
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.mini_batch_size = mini_batch_size
        self.epochs = epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.c1 = c1  # Value function coefficient
        self.c2 = c2  # Entropy coefficient
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, eps=1e-5)
        self.replay_buffer = PrioritizedReplayBuffer(capacity=10000)
        
        # Logging
        self.policy_losses = []
        self.value_losses = []
        self.entropies = []
        self.kl_divs = []

        self.debug = 0

    def select_action(self, state):
        state = torch.from_numpy(state).float().to(self.device)
        with torch.no_grad():
            action_logits, state_value = self.model(state, return_value=True)
            action_probs = torch.softmax(action_logits, dim=-1)

            mask = self.env.get_action_mask(self.device)
            action_probs = action_probs * mask
            action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True)

            action_distribution = torch.distributions.Categorical(action_probs)
            action = action_distribution.sample()
            log_prob = action_distribution.log_prob(action)

        return action.item(), log_prob.item(), state_value.item()

    def store(self, state, action, reward, next_state, done, log_prob, value):
        experience = (state, action, reward, next_state, done, log_prob, value)
        self.replay_buffer.add(experience)

    def update(self, states, actions, rewards, next_states, dones, log_probs, values):
        stime = debug_start(self.debug, 'update')
        advantages, returns = self.compute_advantages_and_returns(rewards, values, dones)

        states = torch.from_numpy(np.array(states)).float().to(self.device)
        actions = torch.from_numpy(np.array(actions)).long().to(self.device)
        old_log_probs = torch.from_numpy(np.array(log_probs)).float().to(self.device)
        advantages = torch.from_numpy(advantages).float().to(self.device)
        returns = torch.from_numpy(returns).float().to(self.device)
        old_values = torch.from_numpy(np.array(values)).float().to(self.device)

        states = states.squeeze(1)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        action_logits, state_values = self.model(states, return_value=True)
        new_log_probs = torch.distributions.Categorical(torch.softmax(action_logits, dim=-1)).log_prob(actions)
        entropy = torch.distributions.Categorical(torch.softmax(action_logits, dim=-1)).entropy().mean()

        ratio = torch.exp(new_log_probs - old_log_probs)
        surrogate1 = ratio * advantages
        surrogate2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surrogate1, surrogate2).mean()

        values_clipped = old_values + (state_values - old_values).clamp(-self.clip_epsilon, self.clip_epsilon)
        value_loss1 = (state_values - returns).pow(2)
        value_loss2 = (values_clipped - returns).pow(2)
        value_loss = 0.5 * torch.max(value_loss1, value_loss2).mean()

        loss = policy_loss + self.c1 * value_loss - self.c2 * entropy

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
        self.optimizer.step()

        # Logging
        self.policy_losses.append(policy_loss.item())
        self.value_losses.append(value_loss.item())
        self.entropies.append(entropy.item())
        approx_kl_div = ((old_log_probs - new_log_probs) ** 2).mean().item()
        self.kl_divs.append(approx_kl_div)
        debug_end(stime)

    def compute_advantages_and_returns(self, rewards, values, dones):
        advantages = np.zeros_like(rewards)
        returns = np.zeros_like(rewards)
        last_gae_lam = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value = 0
            else:
                next_non_terminal = 1.0 - dones[t + 1]
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            advantages[t] = last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam

        # Handle the last timestep separately
        advantages[-1] = rewards[-1] - values[-1]
        
        returns = advantages + values
        return advantages, returns

    def replay_buffer_update(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones, log_probs, values, _, _ = self.replay_buffer.sample(self.batch_size)
        self.update(states, actions, rewards, next_states, dones, log_probs, values)

    def adjust_learning_rate(self, step, total_steps):
        lr = 3e-4 * (1 - step / total_steps)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_logs(self):
        logs = {
            'policy_loss': np.mean(self.policy_losses),
            'value_loss': np.mean(self.value_losses),
            'entropy': np.mean(self.entropies),
            'approx_kl_div': np.mean(self.kl_divs)
        }
        self.policy_losses = []
        self.value_losses = []
        self.entropies = []
        self.kl_divs = []
        return logs