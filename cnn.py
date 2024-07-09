import torch
import torch.nn as nn

class CNNLSTMModel(nn.Module):
    def __init__(self, height, width, depth, nt):
        super(CNNLSTMModel, self).__init__()
        self.conv3d_1 = nn.Conv3d(1, 32, (1, 3, 3))
        self.conv3d_2 = nn.Conv3d(32, 64, (1, 1, 1))
        self.conv3d_3 = nn.Conv3d(64, 128, (1, 2, 2))
        self.relu = nn.ReLU()

        self.flatten = nn.Flatten()
        self.repeat_vector = lambda x: x.unsqueeze(1)  # Custom function to replicate RepeatVector(1) behavior

        self.hiddens = 512
        self.lstm = nn.LSTM(input_size=128*height*27, hidden_size=self.hiddens, batch_first=True)
        self.dense_1 = nn.Linear(512, 512)
        self.dense_2 = nn.Linear(512, 5)
        

    def forward(self, x):
        # Ensure input is float32
        x = x.float()
        # print(f"input state shape: {x.shape}")

        # Convolutional layers
        x = self.conv3d_1(x)
        x = self.relu(x)
        # print(f"conv 1: {x.shape}")
        x = self.conv3d_2(x)
        x = self.relu(x)
        # print(f"conv 2: {x.shape}")
        x = self.conv3d_3(x)
        x = self.relu(x)
        # print(f"after convs: {x.shape}") 

        # Flatten and repeat vector for LSTM
        x = self.flatten(x)
        # print(f"after flatten: {x.shape}")
        x = self.repeat_vector(x)
        # print(f"extend dim: {x.shape}")

        # LSTM layer
        lstm_out, _ = self.lstm(x)
        
        # Flatten output from LSTM
        lstm_out = lstm_out.contiguous().view(-1, lstm_out.shape[-1])
        # print(f"lstm out: {lstm_out.shape}")
        
        # Dense layers
        x = self.dense_1(lstm_out)
        # print(f"first dense: {x.shape}")
        x = torch.relu(x)
        x = self.dense_2(x)
        # print(f"last dense - model out: {x.shape}")
        
        return x
