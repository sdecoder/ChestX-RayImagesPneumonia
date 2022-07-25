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

import argparse

import onnx
import tensorrt as trt
import os
import torch
import torchvision
import pycuda.driver as cuda
import pycuda.autoinit
from PIL import Image
import numpy as np
import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from enum import Enum

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


def build_engine_common_routine(network, builder, config, runtime, engine_file_path):

  plan = builder.build_serialized_network(network, config)
  if plan == None:
    print("[trace] builder.build_serialized_network failed, exit -1")
    exit(-1)
  engine = runtime.deserialize_cuda_engine(plan)
  print("[trace] Completed creating Engine")
  with open(engine_file_path, "wb") as f:
    f.write(plan)
  return engine


def export2onnx(model, onnx_file_name, input_shape):
  print(f'[trace] working in the func: create_onnx_file')
  if os.path.exists(onnx_file_name):
    print(f'[trace] {onnx_file_name} exist, return')
    return onnx_file_name

  print(f'[trace] start to export the torchvision resnet50')
  input_name = ['input']
  output_name = ['output']
  from torch.autograd import Variable

  '''
  import torch.nn as nn
  # model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
  model.fc = nn.Linear(2048, 10, bias=True)
  '''

  # Input to the model
  # [trace] input shape: torch.Size([16, 3, 512, 512])
  '''
  batch_size = 16
  channel = 3
  pic_dim = 512

  '''

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  input = torch.randn(input_shape, requires_grad=True)
  input = input.to(device)
  '''
  output = model(input)
  print(f'[trace] model output: {output.size()}')
    dynamic_axes = {
    'input': {0: 'batch_size'}
  }
  '''

  # Export the model
  torch.onnx.export(model,  # model being run
                    input,  # model input (or a tuple for multiple inputs)
                    onnx_file_name,  # where to save the model (can be a file or file-like object)
                    export_params=True,  # store the trained parameter weights inside the model file
                    opset_version=16,  # the ONNX version to export the model to
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names=['input'],  # the model's input names
                    output_names=['output'],  # the model's output names
                    dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                                  'output': {0: 'batch_size'}})

  print(f'[trace] done with onnx file exporting')
  # modify the network to adapat MNIST
  pass


class CalibratorMode(Enum):
  INT8 = 0
  FP16 = 1
  TF32 = 2
  FP32 = 3


class Calibrator(trt.IInt8EntropyCalibrator2):

  def __init__(self, training_loader, cache_file, element_bytes, batch_size=16, ):
    # Whenever you specify a custom constructor for a TensorRT class,
    # you MUST call the constructor of the parent explicitly.
    trt.IInt8EntropyCalibrator2.__init__(self)
    self.cache_file = cache_file
    self.data_provider = training_loader
    self.batch_size = batch_size
    self.current_index = 0

    # we assume single element is 4 byte
    mem_size = element_bytes * batch_size
    self.device_input = cuda.mem_alloc(mem_size)

  def get_batch_size(self):
    return self.batch_size

  # TensorRT passes along the names of the engine bindings to the get_batch function.
  # You don't necessarily have to use them, but they can be useful to understand the order of
  # the inputs. The bindings list is expected to have the same ordering as 'names'.
  def get_batch(self, names):

    max_data_item = len(self.data_provider)
    if self.current_index + self.batch_size > max_data_item:
      return None

    imgs, labels = next(iter(self.data_provider))
    _elements = imgs.ravel().numpy()
    cuda.memcpy_htod(self.device_input, _elements)
    self.current_index += self.batch_size
    return [self.device_input]

  def read_calibration_cache(self):
    # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
    print(f'[trace] Calibrator: read_calibration_cache: {self.cache_file}')
    if os.path.exists(self.cache_file):
      with open(self.cache_file, "rb") as f:
        return f.read()

  def write_calibration_cache(self, cache):
    print(f'[trace] Calibrator: write_calibration_cache: {cache}')
    with open(self.cache_file, "wb") as f:
      f.write(cache)


def GiB(val):
  return val * 1 << 30
