
###loading
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader,random_split

class BidirectionalGRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BidirectionalGRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=2, bidirectional=True,batch_first=True)
#         self.linears = nn.ModuleList([nn.Linear(hidden_size, 1) for _ in range(48)])
        self.fc2 = nn.Tanh()
        self.fc1 = nn.Linear(hidden_size*2,1)

    def forward(self, x):
        # Pass the input through the GRU layer
        output, _ = self.gru(x)  # Output shape: (batch_size, sequence_length, hidden_size)
        output = self.fc1(output)
        output = self.fc2(output)
        return output


