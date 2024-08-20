# PER.py

import numpy as np

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.capacity = capacity
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.alpha = alpha  # Controls the prioritization strength
        self.beta = beta  # Controls the importance sampling correction
        self.beta_increment = beta_increment

    def add(self, experience):
        max_priority = self.priorities.max() if len(self.buffer) > 0 else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.pos] = experience

        self.priorities[self.pos] = max_priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        N = len(self.buffer)
        if N == 0:
            # Return empty arrays for each expected component
            return [], [], [], [], [], [], [], [], []

        priorities = self.priorities[:N]
        probs = priorities ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(N, batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        # Importance sampling weights
        weights = (N * probs[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize weights
        self.beta = min(1.0, self.beta + self.beta_increment)

        # Determine tuple length and unpack accordingly
        tuple_length = len(samples[0])
        if tuple_length == 7:  # PPO tuple
            states, actions, rewards, next_states, dones, log_probs, values = zip(*samples)
            return (
                np.array(states),
                np.array(actions),
                np.array(rewards),
                np.array(next_states),
                np.array(dones),
                np.array(log_probs),
                np.array(values),
                indices,
                weights,
            )
        elif tuple_length == 5:  # DQN tuple
            states, actions, rewards, next_states, dones = zip(*samples)
            return (
                np.array(states),
                np.array(actions),
                np.array(rewards),
                np.array(next_states),
                np.array(dones),
                indices,
                weights,
            )
        else:
            raise ValueError("Unexpected experience tuple length.")

    def update_priorities(self, indices, errors):
        for idx, error in zip(indices, errors):
            self.priorities[idx] = abs(error) + 1e-5  # Add a small constant to avoid zero priorities

    def __len__(self):
        return len(self.buffer)