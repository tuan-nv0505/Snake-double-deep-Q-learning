from torch import nn
import torch

class DeepQNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.cv1 = nn.Sequential(nn.Conv2d(3, 16, 3, 2), nn.ReLU())
        self.cv2 = nn.Sequential(nn.Conv2d(16, 32, 3, 2), nn.ReLU())
        self.cv3 = nn.Sequential(nn.Conv2d(32, 32, 3, 1), nn.ReLU())

        self.fc = nn.Sequential(
            nn.Linear(32 * 4 * 8 + 8, 256),
            nn.ReLU(),
            nn.Linear(256, 3)
        )

        self.__create_weights()

    def __create_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -0.01, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, state_space, state_logic):
        """
        :param state_space: (batch, 3, 30, 46)
        :param state_logic: (batch, 8)
        :return: Q_values (batch, 3)
        """
        output = self.cv1(state_space)
        output = self.cv2(output)
        output = self.cv3(output)
        output = output.view(output.shape[0], -1)
        output = torch.cat((output, state_logic), dim=1)
        output = self.fc(output)

        return output




