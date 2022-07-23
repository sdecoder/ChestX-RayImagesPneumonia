import os
import sys
import glob
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm

import torch
import torchvision.transforms
from torchvision import transforms, datasets

import tensorrt as trt
from PIL import Image
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import time
import os
import cv2
import torch
import argparse

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


TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def data_transforms(phase=None):

  data_T = T.Compose([

    T.Resize(size=(224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
  ])
  return data_T


def test(trtengine):

  print(f'[trace] start to run on test data')
  test_dataset_normal_path = '../data/chest_xray/test'
  testset = datasets.ImageFolder(os.path.join(test_dataset_normal_path), transform=data_transforms())
  testloader = DataLoader(testset, batch_size=64, shuffle=True)
  print(f"[trace] len of testloader: {len(testloader)}")
  criterion = nn.CrossEntropyLoss()
  running_loss = 0
  if torch.cuda.is_available():
    criterion = criterion.cuda()
    for images, labels in testloader:
      if torch.cuda.is_available():
        images = images.cuda()
        labels = labels.cuda()
        output = model(images)
        loss = criterion(output, labels)
        running_loss += loss.item()

  #loss = criterion(output, target)
  print(f'[trace] final run loss: {running_loss}')
  pass

class HostDeviceMem(object):
  def __init__(self, host_mem, device_mem):
    self.host = host_mem
    self.device = device_mem

  def __str__(self):
    return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

  def __repr__(self):
    return self.__str__()

def allocate_buffers_for_encoder(engine):
  """Allocates host and device buffer for TRT engine inference.
  This function is similair to the one in common.py, but
  converts network outputs (which are np.float32) appropriately
  before writing them to Python buffer. This is needed, since
  TensorRT plugins doesn't support output type description, and
  in our particular case, we use NMS plugin as network output.
  Args:
      engine (trt.ICudaEngine): TensorRT engine
  Returns:
      inputs [HostDeviceMem]: engine input memory
      outputs [HostDeviceMem]: engine output memory
      bindings [int]: buffer to device bindings
      stream (cuda.Stream): cuda stream for engine inference synchronization
  """
  print('[trace] reach func@allocate_buffers')
  inputs = []
  outputs = []
  bindings = []
  stream = cuda.Stream()

  binding_to_type = {}
  binding_to_type['input'] = np.float32
  binding_to_type['output'] = np.float32

  # Current NMS implementation in TRT only supports DataType.FLOAT but
  # it may change in the future, which could brake this sample here
  # when using lower precision [e.g. NMS output would not be np.float32
  # anymore, even though this is assumed in binding_to_type]


  for binding in engine:
    print(f'[trace] current binding: {str(binding)}')
    _binding_shape = engine.get_binding_shape(binding)
    _volume = trt.volume(_binding_shape)
    size = _volume * engine.max_batch_size
    print(f'[trace] current binding size: {size}')
    dtype = binding_to_type[str(binding)]
    # Allocate host and device buffers
    host_mem = cuda.pagelocked_empty(size, dtype)
    device_mem = cuda.mem_alloc(host_mem.nbytes)
    # Append the device buffer to device bindings.
    bindings.append(int(device_mem))
    # Append to the appropriate list.
    if engine.binding_is_input(binding):
      inputs.append(HostDeviceMem(host_mem, device_mem))
    else:
      outputs.append(HostDeviceMem(host_mem, device_mem))
  return inputs, outputs, bindings, stream


def test_using_trt():

  trt.init_libnvinfer_plugins(TRT_LOGGER, '')
  # Initialize runtime needed for loading TensorRT engine from file
  trt_runtime = trt.Runtime(TRT_LOGGER)
  # TRT engine placeholder
  trt_engine = None
  trt_engine_path = './classify-sim.engine'
  if (os.path.exists(trt_engine_path) == False):
    print(f'[trace] engine file {trt_engine_path} does not exist, exit')
    exit(-1)

  with open(trt_engine_path, 'rb') as f:
    engine_data = f.read()
  engine = trt_runtime.deserialize_cuda_engine(engine_data)
  print("[trace] TensorRT engine loaded")
  print("[trace] allocating buffers for TensorRT engine")
  inputs, outputs, bindings, stream = allocate_buffers_for_encoder(trt_engine)

  pass

def main():

  print("[trace] reach the main entry")

  '''
  infer_with_torch_network()
  return
  '''


  print(f'[trace] end of the main point')
  pass

if __name__ == "__main__":
  main()
  pass
