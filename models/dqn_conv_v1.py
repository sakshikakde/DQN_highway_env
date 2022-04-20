import sys
from pathlib import Path
sys.path.append(str(Path( __file__ ).parent.joinpath('..')))
from common_utils import *

class DQN(nn.Module):
    def __init__(self, img_height, img_width, img_stack = 4, num_actions = 2):
        super().__init__()

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(p = 0.2)

        self.conv1 = nn.Conv2d(4, 32, 3, padding='valid')
        self.conv2 = nn.Conv2d(32, 32, 3, padding='valid')
        self.conv3 = nn.Conv2d(32, 64, 3, padding='valid')
        
        self.conv4 = nn.Conv2d(64, 64, 3, padding='valid')
        self.conv5 = nn.Conv2d(64, 128, 3, padding='valid')

        self.conv6 = nn.Conv2d(128, 128, 3, padding='valid')
        self.conv7 = nn.Conv2d(128, 256, 3, padding='valid')
        self.conv8 = nn.Conv2d(256, 512, 3, padding='valid')

        self.fc1 = nn.Linear(13824, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, out_features=num_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.dropout(self.pool(x))

        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.dropout(self.pool(x))

        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = self.dropout(self.pool(x))

        x = F.relu(self.conv8(x))
        x = self.pool(x)

        x = torch.flatten(x, 1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x