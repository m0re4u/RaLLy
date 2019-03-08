import torch.nn as nn
import torch

class DQN(nn.Module):
    def __init__(self, num_inputs=21, num_control_actions=5):
        """
        """
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(num_inputs, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64,num_control_actions)
        self.fc3_1 = nn.Linear(64, 1)
        self.fc3_2 = nn.Linear(64, 1)
        self.fc3_3 = nn.Linear(64, 1)
        self.soft = nn.Softmax()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        control = torch.sigmoid(self.fc3(x))
        jump = self.soft(self.fc3_1(x))
        boost = self.soft(self.fc3_2(x))
        handbrake = self.soft(self.fc3_3(x))

        return control, jump, boost, handbrake