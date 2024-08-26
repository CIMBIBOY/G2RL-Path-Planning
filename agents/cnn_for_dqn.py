import torch
import torch.nn as nn
import numpy as np

class CNNLSTMActor(nn.Module):
    def __init__(self, height=15, width=15, nt=7, nc=3, dropout_rate=0.2):
        super(CNNLSTMActor, self).__init__()
        self.nc = nc
        self.conv_blocks = nn.ModuleList()
        
        in_channels = 4
        out_channels = 32
        for _ in range(nc):
            self.conv_blocks.append(nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=(2, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
                # nn.BatchNorm3d(out_channels),
                nn.ReLU(),
                # nn.Dropout3d(dropout_rate),
                nn.Conv3d(out_channels, out_channels, kernel_size=(2, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
                # nn.BatchNorm3d(out_channels),
                nn.ReLU(),
                # nn.Dropout3d(dropout_rate)
            ))
            in_channels = out_channels
            out_channels *= 2
        
        self.lstm_input_size = 128 * 2 * 2 * 4
        
        self.lstm = nn.LSTM(input_size=self.lstm_input_size, hidden_size=512, batch_first=False)
        
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 5)  # Action logits output
        
        self.dropout = nn.Dropout(dropout_rate)
        
        self.apply(self.orthogonal_init)

        self.debug = True

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
        # print(f"Shape after lstm block: {x.shape}")
        
        x = self.fc1(new_hidden)
        x = torch.relu(x)
        # x = self.dropout(x)
        # print(f"Shape after first linear + relu: {x.shape}")
        
        action_logits = self.fc2(x)
        # print(f"Shape after second linear + relu: {x.shape}")
        return action_logits
    
    def orthogonal_init(self, module):
        if isinstance(module, (nn.Conv3d, nn.Linear)):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    nn.init.orthogonal_(param, gain=np.sqrt(2))
                elif 'bias' in name:
                    nn.init.constant_(param, 0)
        elif isinstance(module, nn.BatchNorm3d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)

        # Recursively initialize submodules
        for child in module.children():
            self.orthogonal_init(child)