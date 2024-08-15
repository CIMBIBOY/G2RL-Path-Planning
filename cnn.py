import torch
import torch.nn as nn
import numpy as np

class CNNLSTMActor(nn.Module):
    def __init__(self, height=15, width=15, nt=4, nc=3, dropout_rate=0.2):
        super(CNNLSTMActor, self).__init__()
        self.nc = nc
        self.conv_blocks = nn.ModuleList()
        
        in_channels = nt
        out_channels = 32
        for _ in range(nc):
            self.conv_blocks.append(nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(),
                nn.Dropout3d(dropout_rate),
                nn.Conv3d(out_channels, out_channels, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(),
                nn.Dropout3d(dropout_rate)
            ))
            in_channels = out_channels
            out_channels *= 2
        
        self.lstm_input_size = 128 * 2 * 2 * 4
        
        self.lstm = nn.LSTM(input_size=self.lstm_input_size, hidden_size=512, batch_first=True)
        
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 5)  # Action logits output
        
        self.dropout = nn.Dropout(dropout_rate)
        
        self.apply(self.orthogonal_init)

    def forward(self, x):
        # print(f"Input shape before processing: {x.shape}")
        batch_size, nt, height, width, channels = x.shape

        x = x.permute(0, 4, 1, 2, 3).contiguous()  # Change to (batch_size, nt, channels, height, width)
        # print(f"Input shape after permute: {x.shape}")
        
        for conv_block in self.conv_blocks:
            x = conv_block(x)
        # print(f"Shape after conv block: {x.shape}")
        
        # Flatten the spatial dimensions and combine with the time dimension
        _, c, t, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(batch_size, t, -1)
        # print(f"Shape after flatten: {x.shape}")
        
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]  # Take the last output
        # print(f"Shape after lstm block: {x.shape}")
        
        x = self.fc1(lstm_out)
        x = torch.relu(x)
        x = self.dropout(x)
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


class CNNLSTMValue(nn.Module):
    def __init__(self, height=15, width=15, nt=4, nc=3, dropout_rate=0.2):
        super(CNNLSTMValue, self).__init__()
        self.nc = nc
        self.conv_blocks = nn.ModuleList()
        
        in_channels = nt
        out_channels = 32
        for _ in range(nc):
            self.conv_blocks.append(nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(),
                nn.Dropout3d(dropout_rate),
                nn.Conv3d(out_channels, out_channels, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(),
                nn.Dropout3d(dropout_rate)
            ))
            in_channels = out_channels
            out_channels *= 2
        
        self.lstm_input_size = 128 * 2 * 2 * 4
        
        self.lstm = nn.LSTM(input_size=self.lstm_input_size, hidden_size=512, batch_first=True)
        
        self.fc1 = nn.Linear(512, 512)
        self.value_head = nn.Linear(512, 1)  # Value output
        
        self.dropout = nn.Dropout(dropout_rate)
        
        self.apply(self.orthogonal_init)

    def forward(self, x):
        batch_size, nt, height, width, channels = x.shape

        x = x.permute(0, 4, 1, 2, 3).contiguous()  # Change to (batch_size, nt, channels, height, width)
        
        for conv_block in self.conv_blocks:
            x = conv_block(x)
        
        _, c, t, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(batch_size, t, -1)
        
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]  # Take the last output
        
        x = self.fc1(lstm_out)
        x = torch.relu(x)
        x = self.dropout(x)
        
        value = self.value_head(x).squeeze(-1)  # Value output
        return value
    
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

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, envs):
        super(Agent, self).__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(1, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(512, 128)
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)
        self.actor = layer_init(nn.Linear(128, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(128, 1), std=1)

    def get_states(self, x, lstm_state, done):
        hidden = self.network(x / 255.0)

        # LSTM logic
        batch_size = lstm_state[0].shape[1]
        hidden = hidden.reshape((-1, batch_size, self.lstm.input_size))
        done = done.reshape((-1, batch_size))
        new_hidden = []
        for h, d in zip(hidden, done):
            h, lstm_state = self.lstm(
                h.unsqueeze(0),
                (
                    (1.0 - d).view(1, -1, 1) * lstm_state[0],
                    (1.0 - d).view(1, -1, 1) * lstm_state[1],
                ),
            )
            new_hidden += [h]
        new_hidden = torch.flatten(torch.cat(new_hidden), 0, 1)
        return new_hidden, lstm_state

    def get_value(self, x, lstm_state, done):
        hidden, _ = self.get_states(x, lstm_state, done)
        return self.critic(hidden)

    def get_action_and_value(self, x, lstm_state, done, action=None):
        hidden, lstm_state = self.get_states(x, lstm_state, done)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden), lstm_state