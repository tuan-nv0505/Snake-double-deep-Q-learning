from torch import nn
import torch

class DeepQNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Sequential(nn.Linear(9, 128), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(128, 256), nn.ReLU())
        self.fc3 = nn.Sequential(nn.Linear(256, 3))


    def forward(self, input):
        output = self.fc1(input)
        output = self.fc2(output)
        output = self.fc3(output)

        return output




