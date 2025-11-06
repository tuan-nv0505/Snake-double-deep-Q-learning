from torch import nn
import torch

class DeepQNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Sequential(nn.Linear(11, 128), nn.LeakyReLU())
        self.fc2 = nn.Sequential(nn.Linear(128, 3))


    def forward(self, input):
        output = self.fc1(input)
        output = self.fc2(output)

        return output




