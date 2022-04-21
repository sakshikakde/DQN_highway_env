import os
import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T   
import highway_env
import time
import pandas as pd

from experience import *
from replay_mem import *
from strategy import *
from agent import *
from env_manager import *
from q_value import *
from opts import *


def plot(values, moving_avg_period, opt, title = 'Training', y_label = 'Duration', name = 'plots/duration.png'):
    plt.figure()
    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel(y_label)
    plt.plot(values)
    moving_avg = get_moving_average(moving_avg_period, values)
    plt.plot(moving_avg)
    plt.legend(['Actual vales', 'Averaged values']) 
    if opt is None:
        save_folder = './snapshots'
    else:
        save_folder = opt.save_folder
    plt.savefig(os.path.join(save_folder, name))
    plt.show()


def get_moving_average(period, values):
    values = torch.tensor(values, dtype=torch.float)
    if len(values) >= period:
        moving_avg = values.unfold(dimension=0, size=period, step=1) \
            .mean(dim=1).flatten(start_dim=0)
        # moving_avg = torch.cat((torch.zeros(period-1), moving_avg))
        return moving_avg.numpy()
    else:
        moving_avg = values.unfold(dimension=0, size=len(values), step=1) \
            .mean(dim=1).flatten(start_dim=0)
        # moving_avg = torch.cat((torch.zeros(len(values)-1), moving_avg))
        return moving_avg.numpy()

def extract_tensors(experiences):
    # Convert batch of Experiences to Experience of batches
    batch = Experience(*zip(*experiences))

    t1 = torch.cat(batch.state)
    t2 = torch.cat(batch.action)
    t3 = torch.cat(batch.reward)
    t4 = torch.cat(batch.next_state)

    return (t1,t2,t3,t4)

def write2csv(filename, duration, reward, loss):
    data = pd.DataFrame([[duration, reward, loss]])
    data.to_csv(filename,  mode = 'a', index=False, header = False)