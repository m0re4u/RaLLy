import torch.nn as nn
import torch

class Actor(nn.Module):
    def __init__(self, num_inputs=9, num_control_actions=1):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(num_inputs, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64,num_control_actions)
        self.fc3_1 = nn.Linear(64, 1)
        self.fc3_2 = nn.Linear(64, 1)
        self.fc3_3 = nn.Linear(64, 1)
        self.soft = nn.Softmax(dim=0)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        control = torch.tanh(self.fc3(x))
        # jump = self.soft(self.fc3_1(x))
        # boost = self.soft(self.fc3_2(x))
        # handbrake = self.soft(self.fc3_3(x))
        jump = 0 * self.fc3_1(x)
        boost = 0 * self.fc3_2(x)
        handbrake = 0 * self.fc3_3(x)

        return control, jump, boost, handbrake

class Critic(nn.Module):
    def __init__(self, num_inputs=9, num_actions=4):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(num_inputs, 128)
        self.fc2 = nn.Linear(128 + num_actions, 64)
        self.fc3 = nn.Linear(64,32)
        self.fc4 = nn.Linear(32,1)

    def forward(self, x, u):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(torch.cat([x, u], 1)))
        q = self.fc3(x)
        q = self.fc4(q)

        return q
