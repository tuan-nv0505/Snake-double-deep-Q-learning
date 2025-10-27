from torch import nn

class DeepQNetwork(nn.Module):
    def __init__(self, action_size):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(4, 32, 8, 4),
            nn.LeakyReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 4, 2),
            nn.LeakyReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1),
            nn.LeakyReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.LeakyReLU(),
            nn.Linear(512, action_size)
        )
        
    def forward(self, input):
        input = self.conv1(input)
        input = self.conv2(input)
        input = self.conv3(input)
        input = input.view(input.shape[0], -1)
        output = self.fc(input)
        return output

