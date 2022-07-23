import numpy as np
import pandas as pd
import torch
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F
from alive_progress import alive_bar
from torch import nn, optim
from torchvision import transforms as T, datasets, models
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from collections import OrderedDict
from tqdm import tqdm

pd.options.plotting.backend = "plotly"
from torch import nn, optim
from torch.autograd import Variable
class classify(nn.Module):
  def __init__(self, num_classes=2):
    super(classify, self).__init__()
    self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)
    self.bn1 = nn.BatchNorm2d(num_features=12)
    self.relu1 = nn.ReLU()
    self.pool = nn.MaxPool2d(kernel_size=2)
    self.conv2 = nn.Conv2d(in_channels=12, out_channels=20, kernel_size=3, stride=1, padding=1)
    self.relu2 = nn.ReLU()
    self.conv3 = nn.Conv2d(in_channels=20, out_channels=32, kernel_size=3, stride=1, padding=1)
    self.bn3 = nn.BatchNorm2d(num_features=32)
    self.relu3 = nn.ReLU()
    self.fc = nn.Linear(in_features=32 * 112 * 112, out_features=num_classes)

    # Feed forwad function

  def forward(self, input):
    output = self.conv1(input)
    output = self.bn1(output)
    output = self.relu1(output)
    output = self.pool(output)
    output = self.conv2(output)
    output = self.relu2(output)
    output = self.conv3(output)
    output = self.bn3(output)
    output = self.relu3(output)
    output = output.view(-1, 32 * 112 * 112)
    output = self.fc(output)

    return output
