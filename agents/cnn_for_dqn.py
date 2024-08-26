import torch
import torch.nn as nn
import numpy as np

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dimension = 7, dropout_rate=0.2):
        super(ConvBlock, self).__init__()
        self.conv1 = layer_init(nn.Conv3d(in_channels, out_channels, kernel_size=(int(time_dimension/3), 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)))
        # self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = layer_init(nn.Conv3d(out_channels, out_channels, kernel_size=(int(time_dimension/3), 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)))
        # self.bn2 = nn.BatchNorm3d(out_channels)
        # self.dropout = nn.Dropout3d(dropout_rate)

    def forward(self, x):
        x = self.conv1(x)
        # x = self.bn1(x)
        x = torch.relu(x)
        # x = self.dropout(x)
        x = self.conv2(x)
        # x = self.bn2(x)
        x = torch.relu(x)
        # x = self.dropout(x)
        return x

class CNNLSTMActor(nn.Module):
    def __init__(self, time_dim = 7):
        super(CNNLSTMActor, self).__init__()
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
        
        self.fc1 = layer_init(nn.Linear(512, 512))
        # self.dropout = nn.Dropout(0.2)
        self.fc2 = layer_init(nn.Linear(512, 5), std=0.01)
        
        self.debug = False

    def forward(self, x, lstm_state, done):
        # print(f"Input shape before processing: {x.shape}")
        batch_size, nt, height, width, channels = x.shape

        x = x.permute(0, 4, 1, 2, 3).contiguous()  # Change to (batch_size, nt, channels, height, width)
        if self.debug:
            print(f"Input shape after permute: {x.shape}")
        
        for conv_block in self.conv_blocks:
            x = conv_block(x / 255.0)
        if self.debug:
            print(f"Shape after conv block: {x.shape}")
        
        hidden = self.flatten(x)
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

        if self.debug:
            print(lstm_state[0].shape)
            print(lstm_state[1].shape)
            print((1.0 - done).view(1, -1, 1).shape)
        
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
        # print(f"Shape after lstm block: {x.shape}")
        
        x = self.fc1(new_hidden)
        x = torch.relu(x)
        # x = self.dropout(x)
        # print(f"Shape after first linear + relu: {x.shape}")
        
        action_logits = self.fc2(x)
        # print(f"Shape after second linear + relu: {x.shape}")
        return action_logits, lstm_state
    