import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        
        self.lstm = nn.LSTM(input_size=input_size, 
                            hidden_size=hidden_size, 
                            num_layers=num_layers, 
                            batch_first=True)
        
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Forward pass through the LSTM layer
        lstm_out, _ = self.lstm(x)
        
        # Only take the output from the final time step
        lstm_out = lstm_out[:, -1, :]
        
        # Fully connected layer
        fc = self.fc(lstm_out)
        output = torch.sigmoid(fc)
        return output

