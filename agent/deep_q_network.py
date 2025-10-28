from torch import nn

class DeepQNetwork(nn.Module):
    def __init__(self, action_size):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(4, 32, 8, 4),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, action_size)
        )
        
    def forward(self, input):
        output = self.conv1(input)
        output = self.conv2(output)
        output = self.conv3(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output

