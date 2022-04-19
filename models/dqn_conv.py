import sys
from pathlib import Path
sys.path.append(str(Path( __file__ ).parent.joinpath('..')))
from common_utils import *

class DQN(nn.Module):
    def __init__(self, img_height, img_width, img_stack = 4, num_actions = 2):
        super().__init__()
        self.fc1 = nn.Linear(in_features=img_height*img_width*img_stack, out_features=24)   
        self.fc2 = nn.Linear(in_features=24, out_features=32)
        self.out = nn.Linear(in_features=32, out_features=num_actions)

    def forward(self, t):
        t = t.flatten(start_dim=1)
        t = F.relu(self.fc1(t))
        t = F.relu(self.fc2(t))
        t = self.out(t)
        return t