from torch import nn
import torch

class DeepQNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.cv1 = nn.Sequential(nn.Conv2d(3, 4, 3, 2), nn.ReLU())
        self.cv2 = nn.Sequential(nn.Conv2d(4, 8, 3, 2), nn.ReLU())
        self.cv3 = nn.Sequential(nn.Conv2d(8, 8, 3, 1), nn.ReLU())

        self.fc_logic = nn.Linear(9, 128)

        self.fc = nn.Sequential(
            nn.Linear(8 * 4 * 8 + 128, 512),
            nn.ReLU(),
            nn.Linear(512, 3)
        )


    def forward(self, state_space, state_logic):
        """
        :param state_space: (batch, 3, 30, 46)
        :param state_logic: (batch, 9)
        :return: Q_values (batch, 3)
        """
        output = self.cv1(state_space)
        output = self.cv2(output)
        output = self.cv3(output)
        output = output.view(output.shape[0], -1)
        output = torch.cat((output, state_logic), dim=1)
        output = self.fc(output)

        return output




