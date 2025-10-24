from torch import nn

class DeepQNetwork(nn.Module):
    def __init__(self, action_size):
        super().__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv_3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=600, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=action_size)
        )

    def forward(self, input):
        output = self.conv_1(input)
        output = self.conv_2(output)
        output = self.conv_3(output)
        output = self.fc(output)
        return output

