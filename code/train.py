import time

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

import nn_structure

pd.options.plotting.backend = "plotly"
from torch import nn, optim
from torch.autograd import Variable

TEST = 'test'
TRAIN = 'train'
VAL = 'val'


def data_transforms(phase=None):
  if phase == TRAIN:

    data_T = T.Compose([

      T.Resize(size=(256, 256)),
      T.RandomRotation(degrees=(-20, +20)),
      T.CenterCrop(size=224),
      T.ToTensor(),
      T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
  elif phase == TEST or phase == VAL:

    data_T = T.Compose([

      T.Resize(size=(224, 224)),
      T.ToTensor(),
      T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

  return data_T


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
  # data_dir = "../input/chest-xray-pneumonia/chest_xray/chest_xray"
  data_dir = '../data/chest_xray'
  trainset = datasets.ImageFolder(os.path.join(data_dir, TRAIN), transform=data_transforms(TRAIN))
  testset = datasets.ImageFolder(os.path.join(data_dir, TEST), transform=data_transforms(TEST))
  validset = datasets.ImageFolder(os.path.join(data_dir, VAL), transform=data_transforms(VAL))

  class_names = trainset.classes
  print(class_names)
  print(trainset.class_to_idx)

  print(f"[trace] prepare for the dataloader[train/valid/test]")
  trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
  validloader = DataLoader(validset, batch_size=64, shuffle=True)
  testloader = DataLoader(testset, batch_size=64, shuffle=True)

  images, labels = iter(trainloader).next()
  print(images.shape)
  print(labels.shape)

  '''
    for i, (images, labels) in enumerate(trainloader):
    if torch.cuda.is_available():
      images = Variable(images.cuda())
      labels = Variable(labels.cuda())
  '''

  print(f"[trace] prepare model to train")
  print(f"[trace] current model summary:")
  from torchsummary import summary
  summary(nn_structure.classify().cuda(), (images.shape[1], images.shape[2], images.shape[3]))

  model = nn_structure.classify()
  # defining the optimizer
  optimizer = optim.Adam(model.parameters(), lr=0.01)
  # defining the loss function
  criterion = nn.CrossEntropyLoss()
  # checking if GPU is available
  if torch.cuda.is_available():
    print(f"[trace] cuda available, train using cuda")
    model = model.cuda()
    criterion = criterion.cuda()

    Losses = []
    epochs = 20
    '''
        letters = [chr(ord('A') + x) for x in range(26)]
        with alive_bar(26, dual_line=True, title='Alphabet') as bar:
          for c in letters:
            bar()
            bar.text = f'-> Teaching the letter: {c}, please wait...'
            if c in 'HKWZ':
              print(f'fail "{c}", retry later')
            time.sleep(0.3)      
        exit(-1)
    
    '''

    best_loss = np.inf
    best_model_name = None

    with alive_bar(epochs, dual_line=True, title='Epochs', force_tty = True) as bar:
      for i in range(epochs):

        bar()
        running_loss = 0
        for images, labels in trainloader:

          # Changing images to cuda for gpu

          if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()

          # Training pass
          # Sets the gradient to zero
          optimizer.zero_grad()

          output = model(images)
          loss = criterion(output, labels)

          # This is where the model learns by backpropagating
          # accumulates the loss for mini batch
          loss.backward()

          # And optimizes its weights here
          optimizer.step()
          Losses.append(loss)
          running_loss += loss.item()

          # save the best mode
        else:

          bar.text = "Epoch {} - Training loss: {}".format(i + 1, running_loss / len(trainloader))
          if running_loss < best_loss:
            best_loss = running_loss
            best_epoch = i
            if best_model_name is not None:
              os.remove(best_model_name)

            best_model_name = f"model_{best_epoch}th_epoch_{running_loss}_loss.pth"
            torch.save(model.state_dict(), best_model_name)

  pass


if __name__ == '__main__':
  main()
  pass
