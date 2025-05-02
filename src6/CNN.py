import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNActionValue(nn.Module):
    def __init__(self, input_channels, num_actions):
        super(CNNActionValue, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=8, stride=4)  # 기존: 16
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)             # 기존: 32
        # conv3 없음
        self.fc1 = nn.Linear(32 * 9 * 9, 256)  # 기존 구조에 맞춰짐
        self.fc2 = nn.Linear(256, num_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


