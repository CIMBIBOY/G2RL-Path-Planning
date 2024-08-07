import torch
import torch.nn as nn
import numpy as np

class CNNLSTMModel(nn.Module):
    def __init__(self, height, width, depth, nt, dropout_rate=0.3):
        super(CNNLSTMModel, self).__init__()
        self.conv3d_1 = nn.Conv3d(4, 32, (1, 3, 3))  # Input channels should be 4
        self.bn1 = nn.BatchNorm3d(32)
        self.conv3d_2 = nn.Conv3d(32, 64, (1, 1, 1))
        self.bn2 = nn.BatchNorm3d(64)
        self.conv3d_3 = nn.Conv3d(64, 128, (1, 2, 2))
        self.bn3 = nn.BatchNorm3d(128)
        self.relu = nn.ReLU()

        # Dropout layer after convolutional layers
        self.conv_dropout = nn.Dropout3d(dropout_rate)

        self.flatten = nn.Flatten()
        self.hiddens = 512

        # Calculate the correct input size for LSTM
        conv1_out = (height - 2, width - 2)
        conv2_out = conv1_out
        conv3_out = (conv2_out[0] - 1, conv2_out[1] - 1)
        lstm_input_size = 128 * conv3_out[0] * conv3_out[1]

        # Dropout in LSTM layer (only for stacked LSTMs)
        self.lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=self.hiddens, batch_first=True)

        self.dense_1 = nn.Linear(512, 512)
        self.dense_dropout = nn.Dropout(dropout_rate)
        self.action_head = nn.Linear(512, 5)
        self.value_head = nn.Linear(512, 1)  # Add a separate head for value prediction

        # Apply orthogonal initialization to the model
        self.apply(self.orthogonal_init)

    def forward(self, x, return_value=False):
        # Ensure input is float32
        x = x.float()

        # Input shape: (batch_size, nt, height, width, channels)
        batch_size, nt, height, width, channels = x.shape
        
        # Reshape to (batch_size * nt, channels, 1, height, width)
        x = x.view(batch_size * nt, channels, 1, height, width)

        # Convolutional layers
        x = self.relu(self.bn1(self.conv3d_1(x)))
        x = self.relu(self.bn2(self.conv3d_2(x)))
        x = self.relu(self.bn3(self.conv3d_3(x)))
        
        # Apply dropout after convolutions
        x = self.conv_dropout(x)
        
        # Flatten and reshape for LSTM
        x = self.flatten(x)
        x = x.view(batch_size, nt, -1)

        # LSTM layer
        lstm_out, _ = self.lstm(x)

        # We only need the last output from LSTM
        lstm_out = lstm_out[:, -1, :]
        
        # Dense layers with dropout
        x = self.dense_1(lstm_out)
        x = torch.relu(x)
        x = self.dense_dropout(x)
        
        # Action and value outputs
        action_logits = self.action_head(x)
        
        # If the flag is set, also return the state value
        if return_value:
            state_value = self.value_head(x).squeeze(-1)  # Ensure the value is a scalar
            return action_logits, state_value
        
        return action_logits

    def orthogonal_init(self, m):
        if isinstance(m, (nn.Conv3d, nn.Linear)):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)