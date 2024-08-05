import torch
import torch.nn as nn

class CNNLSTMModel(nn.Module):
    def __init__(self, height, width, depth, nt):
        super(CNNLSTMModel, self).__init__()
        self.conv3d_1 = nn.Conv3d(4, 32, (1, 3, 3))  # Input channels should be 4
        self.bn1 = nn.BatchNorm3d(32)
        self.conv3d_2 = nn.Conv3d(32, 64, (1, 1, 1))
        self.bn2 = nn.BatchNorm3d(64)
        self.conv3d_3 = nn.Conv3d(64, 128, (1, 2, 2))
        self.bn3 = nn.BatchNorm3d(128)
        self.relu = nn.ReLU()

        self.flatten = nn.Flatten()
        self.hiddens = 512

        # Calculate the correct input size for LSTM
        conv1_out = (height - 2, width - 2)
        conv2_out = conv1_out
        conv3_out = (conv2_out[0] - 1, conv2_out[1] - 1)
        lstm_input_size = 128 * conv3_out[0] * conv3_out[1]

        self.lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=self.hiddens, batch_first=True)
        self.dense_1 = nn.Linear(512, 512)
        self.dense_2 = nn.Linear(512, 5)

    def forward(self, x):
        # Ensure input is float32
        x = x.float()
        # print(f"input state shape: {x.shape}")

        # Input shape: (batch_size, nt, height, width, channels)
        batch_size, nt, height, width, channels = x.shape
        
        # Reshape to (batch_size * nt, channels, 1, height, width)
        x = x.view(batch_size * nt, channels, 1, height, width)

        # Convolutional layers
        x = self.relu(self.bn1(self.conv3d_1(x)))
        x = self.relu(self.bn2(self.conv3d_2(x)))
        x = self.relu(self.bn3(self.conv3d_3(x)))
        
        # Flatten and reshape for LSTM
        x = self.flatten(x)
        x = x.view(batch_size, nt, -1)

        # LSTM layer
        lstm_out, _ = self.lstm(x)
        
        # We only need the last output from LSTM
        lstm_out = lstm_out[:, -1, :]
        
        # Dense layers
        x = self.dense_1(lstm_out)
        x = torch.relu(x)
        x = self.dense_2(x)
        
        return x