import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import numpy as np

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.2):
        super(ConvBlock, self).__init__()
        self.conv1 = layer_init(nn.Conv3d(in_channels, out_channels, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)))
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
    def __init__(self):
        super(CNNLSTM, self).__init__()
        self.conv_blocks = nn.Sequential(
            ConvBlock(4, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
        )
        
        self.flatten = nn.Flatten()
        
        self.lstm = nn.LSTM(128 * 2 * 2 * 4, 512, batch_first=True)
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)
        
        self.fc = layer_init(nn.Linear(512, 512))
        self.dropout = nn.Dropout(0.2)
        self.actor = layer_init(nn.Linear(512, 5), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)

    def get_states(self, x, lstm_state, done):
        debug = False
        batch_size, nt, height, width, channels = x.shape

        if debug:
            print(f"Debug get_states: x tensor shape: {x.shape}, done: {done.shape} ")
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        if debug:
            print(f"x tensor shape after reshape: {x.shape}")

        x = self.conv_blocks(x / 255.0)
        if debug:
            print(f"Tensor shape after conv block: {x.shape}")
        hidden = self.flatten(x)
        if debug:
            print(f"Hidden shape after flatten: {hidden.shape}")
            print(f"Batch_size: {batch_size}")
        # Reshape hidden to be (batch_size, sequence_length, input_size)
        hidden = hidden.reshape(batch_size, nt, self.lstm.input_size)
        if debug:
            print(f"Hidden shape after reshape: {hidden.shape}")
        # Only use the last output of the LSTM (corresponding to the present observation)
        _hidden = hidden[:, -1:, :]
        if debug:
            print(f"Hidden after taking current obs: {_hidden.shape}")
        
        # Reshape done to match the batch and sequence dimensions
        done = done.view(batch_size, -1)
        if debug:
            print(f"Done shape: {done.shape}")

        new_hidden = []
        for h, d in zip(_hidden, done):
            h, lstm_state = self.lstm(
                h.unsqueeze(0),
                (
                    (1.0 - d).view(1, -1, 1) * lstm_state[0],
                    (1.0 - d).view(1, -1, 1) * lstm_state[1],
                ),
            )
            new_hidden += [h]
        new_hidden = torch.cat(new_hidden)  
        if debug:
            print(f"New hidden shape: {new_hidden.shape}")

        # Pass through fully connected layer and dropout
        final_hidden = self.fc(new_hidden)
        final_hidden = self.dropout(final_hidden)
        if debug:
            print(f"Final hidden shape: {final_hidden.shape}")
        return final_hidden, lstm_state

    def get_value(self, x, lstm_state, done):
        hidden, _ = self.get_states(x, lstm_state, done)
        return self.critic(hidden)

    def get_action_and_value(self, x, lstm_state, done, env, device, action=None):
        hidden, lstm_state = self.get_states(x, lstm_state, done)
        logits = self.actor(hidden)

        # Action masking
        mask = env.get_action_mask(device)
        action_logits = logits + (1.0 - mask) * (-1e9)  # Penalize invalid actions with a large negative value

        probs = Categorical(logits=action_logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden), lstm_state