import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import numpy as np

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dimension, dropout_rate=0.2):
        super(ConvBlock, self).__init__()
        self.conv1 = layer_init(nn.Conv3d(in_channels, out_channels, kernel_size=(int(time_dimension/2), 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)))
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = layer_init(nn.Conv3d(out_channels, out_channels, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)))
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.dropout = nn.Dropout3d(dropout_rate)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.dropout(x)
        return x

class CNNLSTM(nn.Module):
    def __init__(self, time_dim = 1):
        super(CNNLSTM, self).__init__()
        self.conv_blocks = nn.Sequential(
            ConvBlock(4, 32, time_dim),
            ConvBlock(32, 64, time_dim),
            ConvBlock(64, 128, time_dim),
        )
        
        self.flatten = nn.Flatten()
        
        self.lstm = nn.LSTM(128 * 2 * 2 * 4, 512, batch_first=False)
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)
        
        self.fc = layer_init(nn.Linear(512, 512))
        self.dropout = nn.Dropout(0.2)
        self.actor = layer_init(nn.Linear(512, 5), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)
        
        self.debug = False

    def get_states(self, x, lstm_state, done):
        # Constructed get_states so it works with concatenated past observations (nt time dimension > 1) - leveraging observation_history 
        batch_size, num_envs, nt, height, width, channels = x.shape

        if self.debug:
            print(f"Debug get_states: x tensor shape: {x.shape}, done: {done.shape} ")
        x = x.permute(0, 1, 2, 5, 3, 4).contiguous()
        if self.debug:
            print(f"x tensor shape after reshape: {x.shape}")
        x = x.view(-1, channels, nt, height, width)  # Flatten batch and num_envs into one dimension
        if self.debug:
            print(f"x tensor shape after flatten: {x.shape}")

        x = self.conv_blocks(x / 255.0)
        if self.debug:
            print(f"Tensor shape after conv block: {x.shape}")

        hidden = self.flatten(x)
        batch_size = lstm_state[0].shape[1]
        if self.debug:
            print(f"Hidden shape after flatten: {hidden.shape}")
            print(f"Batch_size: {batch_size}")

        # Reshape hidden to be (batch_size, sequence_length, input_size)
        hidden = hidden.reshape(-1, batch_size, self.lstm.input_size)
        if self.debug:
            print(f"Hidden shape after reshape: {hidden.shape}")

        # Reshape done to match the batch and sequence dimensions
        done = done.view(-1, batch_size)
        if self.debug:
            print(f"Done shape: {done.shape}")  

        # Initialize a list to store the new hidden states  
        
        new_hidden = []
        for h, d in zip(hidden, done):
            # Process each timestep separately, taking care of done flags
            h, lstm_state = self.lstm(
                h.unsqueeze(0), 
                (
                    (1.0 - d).view(1, -1, 1) * lstm_state[0],  # Slice lstm_state to match batch dimension of h
                    (1.0 - d).view(1, -1, 1) * lstm_state[1],  # Slice lstm_state to match batch dimension of h
                ),
            )
            new_hidden += [h]
        # Concatenate along the batch dimension
        new_hidden = torch.flatten(torch.cat(new_hidden), 0, 1)
        if self.debug:
            print(f"New hidden shape: {new_hidden.shape}")

        # Pass through fully connected layer and dropout
        final_hidden = self.fc(new_hidden)
        final_hidden = self.dropout(final_hidden)
        if self.debug:
            print(f"Final hidden shape: {final_hidden.shape}")
        return final_hidden, lstm_state

    def get_value(self, x, lstm_state, done):
        hidden, _ = self.get_states(x, lstm_state, done)
        return self.critic(hidden)

    def get_action_and_value(self, x, lstm_state, done, env, device, action=None, envs_num = 4):
        # Get action masks for all environments
        if self.debug:
            print(f"Input shape: x tensor: {x.shape}, done shape: {done.shape}")
        if envs_num > 1:
            masks = []
            for i in range(x.shape[1]):  # Iterate over each environment
                mask = env.envs[i].action_mask(device = device)  # Get the action mask for each environment
                masks.append(mask)

            # Stack the masks into a single tensor
            masks = torch.stack(masks)  # Shape: (num_envs, num_actions)
            if self.debug:
                print("Action Mask Matrix:")
                print(masks)
                print(f"Mask shape: {mask.shape}")
        else: masks = env.action_mask(device)

        hidden, lstm_state = self.get_states(x, lstm_state, done)
        if self.debug:
            print(f"Hidden shape: {hidden.shape}")
        logits = self.actor(hidden)
        if self.debug:
            # print("Logits Matrix Before Masking:")
            # print(logits)
            print(f"logits shape: {logits.shape}")

        # Multiply logits by the mask
        masked_logits = logits * masks

        if self.debug:
            print(f"masked logits shape: {masked_logits.shape}")

        # Shift logits to avoid large negative values (optional)
        masked_logits = masked_logits - torch.max(masked_logits, dim=-1, keepdim=True)[0]

        # Apply softmax to get the probability distribution
        log_probs = F.log_softmax(masked_logits, dim=-1)
        probs_distribution= torch.exp(log_probs)

        # Use these probabilities to create the categorical distribution
        probs = Categorical(probs=probs_distribution)

        if action is None:
            action = probs.sample()
            if self.debug:
                print("Actions selected")
                print(action)
                print(f"actions shape: {action.shape}")

        log_prob = probs.log_prob(action)
        # log_prob = torch.log(probs.probs + 1e-8)[torch.arange(probs.probs.shape[0]), action]
        
        return action, log_prob, probs.entropy(), self.critic(hidden), lstm_state