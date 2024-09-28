
###loading
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader,random_split

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=7, stride=1, padding=3),
            nn.LeakyReLU(0.1),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=4, padding=2),
            nn.LeakyReLU(0.1),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=8, padding=1),
            nn.LeakyReLU(0.1),
            nn.Flatten()
        )
        
        # Linear section
        self.encoder_lin = nn.Sequential(
            nn.Linear(103680, 4096),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, 1024)
        )
        
        self.decoder_lin = nn.Sequential(
            nn.Linear(1024, 4096),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, 103680),
            nn.LeakyReLU(0.1)
        )
        
        self.unflatten = nn.Unflatten(1, torch.Size([64,1620]))
        
        

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=3, stride=8, padding=1,output_padding=7),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose1d(in_channels=32, out_channels=16, kernel_size=5, stride=4, padding=2,output_padding=3),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose1d(in_channels=16, out_channels=1, kernel_size=7, stride=1, padding=3)
        )
    
    def signal_preprocess(self,x,mask):
        index=mask!=0
        mean= torch.mean(x[:,:,index],dim=(2))  # Compute mean along the dimensions
        std = torch.std(x[:,:,index], dim=(2))  # Compute standard deviation along the dimensions
        normalized_dataset = (x - mean.unsqueeze(2)) / std.unsqueeze(2)  # Normalize the tensor
        cornormalized_dataset = normalized_dataset * mask.reshape((1,1,-1))
        cornormalized_dataset = normalized_dataset.float()
        return cornormalized_dataset,normalized_dataset,mean,std
    
    def signal_backprocess(self,x,mean,std):
        recover=x*std+mean
        return recover

    def forward(self, x):
        # standardize the input
#         nx,mean,std=self.signal_preprocess(x)
        # Encode the input
        encoded = self.encoder(x)
        encoded = self.encoder_lin(encoded)
        
#         print(encoded.size())
        

        # Decode the encoded representation
        decoded = self.decoder_lin(encoded)
        decoded = self.unflatten(decoded)
        decoded = self.decoder(decoded)

        return decoded

