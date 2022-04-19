import sys
from pathlib import Path
sys.path.append(str(Path( __file__ ).parent.joinpath('..')))
from common_utils import *


class DQN(nn.Module):
    def __init__(self, img_height, img_width, img_stack = 4, num_actions = 2):
        super().__init__()

        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(4, 64, 5, padding='valid')
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 5, padding='valid')
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 3, padding='valid')
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, 3, padding='valid')
        self.bn4 = nn.BatchNorm2d(512)

        self.fc5 = nn.Linear(16896, 1024)
        self.bn5 = nn.BatchNorm1d(1024)
        self.fc1 = nn.Linear(1024, 256)
        self.fc2 = nn.Linear(256, 32)
        self.out = nn.Linear(in_features=32, out_features=num_actions)

    def forward(self, x):
        x = self.pool(self.bn1(F.relu(self.conv1(x))))
        x = self.pool(self.bn2(F.relu(self.conv2(x))))
        x = self.pool(self.bn3(F.relu(self.conv3(x))))
        x = F.relu(self.bn4(self.conv4(x)))
        x = torch.flatten(x, 1) 
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x