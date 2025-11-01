from torch import nn

class DeepQNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Sequential(nn.Linear(8, 32), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(32, 64), nn.ReLU())
        self.fc3 = nn.Sequential(nn.Linear(64, 3))

        self.__create_weights()

    def __create_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -0.01, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, input):
        output = self.fc1(input)
        output = self.fc2(output)
        output = self.fc3(output)

        return output


