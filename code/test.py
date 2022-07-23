import cv2
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
from torch.utils.data import DataLoader, Dataset
from collections import OrderedDict
from tqdm import tqdm
import nn_structure
from os import listdir
from os.path import isfile, join

def data_transforms(phase=None):

  data_T = T.Compose([

    T.Resize(size=(224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
  ])
  return data_T


def test(model):

  print(f'[trace] start to run on test data')
  test_dataset_normal_path = '../data/chest_xray/test'
  testset = datasets.ImageFolder(os.path.join(test_dataset_normal_path), transform=data_transforms())
  testloader = DataLoader(testset, batch_size=64, shuffle=True)
  print(f"[trace] len of testloader: {len(testloader)}")
  criterion = nn.CrossEntropyLoss()
  running_loss = 0
  if torch.cuda.is_available():
    model = model.cuda()
    criterion = criterion.cuda()
    for images, labels in testloader:
      if torch.cuda.is_available():
        images = images.cuda()
        labels = labels.cuda()
        output = model(images)
        loss = criterion(output, labels)
        running_loss += loss.item()

  #loss = criterion(output, target)
  print(f'[trace] final run loss by Pytorch: {running_loss}')
  #[trace] final run loss by Pytorch: 4.637269496917725

  pass

def main():

  model_name = 'model_18th_epoch_8.770064184442163_loss.pth'
  print(f'[trace] working at the main entry')
  model = nn_structure.classify()
  model.load_state_dict(torch.load(model_name))
  model.eval()
  print(f'[trace] model loaded')
  test(model)
  pass

if __name__ == '__main__':
  main()
  pass
