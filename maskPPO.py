import torch
import torch.nn as nn
import torch.optim as optim
from PER import PrioritizedReplayBuffer
import numpy as np
from train_utils import debug_start, debug_end

class MaskPPOAgent:
    def __init__(self, env, actor_model, critic_model, device='cpu', batch_size=32, mini_batch_size=8, epochs=4, gamma=0.99, gae_lambda=0.95, clip_epsilon=0.2, c1=0.5, c2=0.01, lr=3e-4):
        self.env = env
        self.actor_model = actor_model
        self.critic_model = critic_model
        self.device = device
        self.batch_size = batch_size
        self.mini_batch_size = mini_batch_size
        self.epochs = epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.c1 = c1  # Value function coefficient
        self.c2 = c2  # Entropy coefficient
        self.actor_optimizer = optim.Adam(self.actor_model.parameters(), lr=lr, eps=1e-5)
        self.critic_optimizer = optim.Adam(self.critic_model.parameters(), lr=lr, eps=1e-5)
        self.replay_buffer = PrioritizedReplayBuffer(capacity=10000)
        
        # Logging
        self.policy_losses = []
        self.value_losses = []
        self.entropies = []
        self.kl_divs = []

        self.debug = 0

    def select_action(self, state):
        if self.debug:
            print("Debug: Selecting action")
            print(f"Debug: Input state shape: {state.shape}")
        
        state = torch.from_numpy(state).float().to(self.device)
        with torch.no_grad():
            action_logits, state_value = self.model(state)
            action_probs = torch.softmax(action_logits, dim=-1)

            mask = self.env.get_action_mask(self.device)
            if self.debug:
                print(f"Debug: Action probabilities before masking: {action_probs}")
                print(f"Debug: Action mask: {mask}")
            
            action_probs = action_probs * mask
            
            if self.debug:
                print(f"Debug: Action probabilities after masking: {action_probs}")

            if action_probs.sum() > 0:
                action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True)
                if self.debug:
                    print(f"Debug: Action probabilities after probability sum: {action_probs}")
                action_distribution = torch.distributions.Categorical(action_probs)
                if self.debug:
                    print(f"Debug: Action distribution: {action_distribution}")
                action = action_distribution.sample()
                log_prob = action_distribution.log_prob(action)
            else:
                action = torch.tensor(4, device=self.device)
                log_prob = torch.tensor(0.0, device=self.device)

            if self.debug:
                print(f"Debug: Action selected: {action.item()}")
                print(f"Debug: Log probability: {log_prob.item()}")
                print(f"Debug: State value: {state_value.item()}")

        return action.item(), log_prob.item(), state_value.item()

    def store(self, state, action, reward, next_state, done, log_prob, value):
        experience = (state, action, reward, next_state, done, log_prob, value)
        self.replay_buffer.add(experience)

    def update(self, states, actions, rewards, next_states, dones, log_probs, values):
        if self.debug:
            print("Debug: Starting update")

        advantages, returns = self.compute_advantages_and_returns(rewards, values, dones)
        if self.debug:
            print(f"Debug: Advantages: {advantages}, returns: {returns}")

        states = torch.from_numpy(np.array(states)).float().to(self.device)
        actions = torch.from_numpy(np.array(actions)).long().to(self.device)
        old_log_probs = torch.from_numpy(np.array(log_probs)).float().to(self.device)
        advantages = torch.from_numpy(advantages).float().to(self.device)
        returns = torch.from_numpy(returns).float().to(self.device)
        old_values = torch.from_numpy(np.array(values)).float().to(self.device)

        states = states.squeeze(1)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        if self.debug:
            print(f"Debug: Normalized advantages mean: {advantages.mean()}, std: {advantages.std()}")

        new_action_logits = self.actor_model(states)
        new_state_values = self.critic_model(states)

        new_log_probs = torch.distributions.Categorical(torch.softmax(new_action_logits, dim=-1)).log_prob(actions)
        entropy = torch.distributions.Categorical(torch.softmax(new_action_logits, dim=-1)).entropy().mean()

        ratio = torch.exp(new_log_probs - old_log_probs)
        surrogate1 = ratio * advantages
        surrogate2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surrogate1, surrogate2).mean()

        values_clipped = old_values + (new_state_values - old_values).clamp(-self.clip_epsilon, self.clip_epsilon)
        value_loss1 = (new_state_values - returns).pow(2)
        value_loss2 = (values_clipped - returns).pow(2)
        value_loss = 0.5 * torch.max(value_loss1, value_loss)

        total_loss = policy_loss + self.c1 * value_loss - self.c2 * entropy

        if self.debug:
            print(f"Debug: Policy loss: {policy_loss.item()}")
            print(f"Debug: Value loss: {value_loss.item()}")
            print(f"Debug: Entropy: {entropy.item()}")
            print(f"Debug: Total loss: {total_loss.item()}")

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_model.parameters(), max_norm=0.5)
        torch.nn.utils.clip_grad_norm_(self.critic_model.parameters(), max_norm=0.5)
        self.actor_optimizer.step()
        self.critic_optimizer.step()

        # Logging
        self.policy_losses.append(policy_loss.item())
        self.value_losses.append(value_loss.item())
        self.entropies.append(entropy.item())
        approx_kl_div = ((old_log_probs - new_log_probs) ** 2).mean().item()
        self.kl_divs.append(approx_kl_div)

        if self.debug:
            print(f"Debug: Approximate KL divergence: {approx_kl_div}")
            print("Debug: Update completed")

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