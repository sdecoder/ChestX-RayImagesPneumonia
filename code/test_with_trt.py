import os
import sys
import glob
import argparse

import numpy
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm

import torch
import torchvision.transforms
from pycuda.gpuarray import GPUArray
from torchvision import transforms, datasets

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

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
batch_size = 64
class HostDeviceMem(object):
  def __init__(self, host_mem, device_mem):
    self.host = host_mem
    self.device = device_mem

  def __str__(self):
    return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

  def __repr__(self):
    return self.__str__()


def allocate_buffers_for_encoder(engine, batch_size):
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
    #size = _volume * engine.max_batch_size
    size = abs(_volume * batch_size) # dynamic batch size
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

def data_transforms(phase=None):

  data_T = T.Compose([

    T.Resize(size=(224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
  ])
  return data_T

def gpu_array_test(device_mem, batch_size):

  shape = (batch_size, 2)
  gpuarray = GPUArray(gpudata=device_mem, shape=shape, dtype=numpy.float32)
  shape = gpuarray.shape
  dtype = gpuarray.dtype
  out_dtype = torch.float32

  # prepare the output container
  out = torch.zeros(shape, dtype=out_dtype).cuda()
  byte_size = gpuarray.itemsize * gpuarray.size
  pycuda.driver.memcpy_dtod(out.data_ptr(), gpuarray.gpudata, byte_size)
  return out

def test(engine):

  print(f'[trace] start to run on test data')
  batch_size = 64
  test_dataset_normal_path = '../data/chest_xray/test'
  testset = datasets.ImageFolder(os.path.join(test_dataset_normal_path), transform=data_transforms())
  testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)
  print(f"[trace] len of testloader: {len(testloader)}")
  criterion = nn.CrossEntropyLoss().cuda()
  running_loss = 0
  context = engine.create_execution_context()
  inputs, outputs, bindings, stream = allocate_buffers_for_encoder(engine, batch_size)
  context.set_binding_shape(0, (batch_size, 3, 224, 224))

  for images, labels in testloader:
    #images = images.cuda()
    if len(images) != batch_size:
      batch_size = len(images)
      inputs, outputs, bindings, stream = allocate_buffers_for_encoder(engine, batch_size)
      context.set_binding_shape(0, (batch_size, 3, 224, 224))

    np.copyto(inputs[0].host, images.ravel()) # already in GPU
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    #cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    stream.synchronize()
    output = outputs[0]
    result_tensor = gpu_array_test(output.device, batch_size)
    '''
    output = torch.from_numpy(output.host)
    output = output.reshape((batch_size, 2))
    loss = criterion(output, labels)
    '''
    labels = labels.cuda()
    loss = criterion(result_tensor, labels)
    running_loss += loss.item()

  #loss = criterion(output, target)
  print(f'[trace] final run loss by TensorRT: {running_loss}')
  # [trace] final run loss by TensorRT: 4.696431130170822
  # [trace] final run loss by TensorRT: 4.671649992465973 *new!
  pass


def test_using_trt():

  trt.init_libnvinfer_plugins(TRT_LOGGER, '')
  # Initialize runtime needed for loading TensorRT engine from file
  trt_runtime = trt.Runtime(TRT_LOGGER)
  # TRT engine placeholder
  trt_engine_path = './classify-sim.engine'
  if (os.path.exists(trt_engine_path) == False):
    print(f'[trace] engine file {trt_engine_path} does not exist, exit')
    exit(-1)

  with open(trt_engine_path, 'rb') as f:
    engine_data = f.read()
  trt_engine = trt_runtime.deserialize_cuda_engine(engine_data)
  print("[trace] TensorRT engine loaded")
  print("[trace] allocating buffers for TensorRT engine")
  test(trt_engine)
  pass

def main():

  print("[trace] reach the main entry")
  test_using_trt()
  print(f'[trace] end of the main point')
  pass

if __name__ == "__main__":
  main()
  pass
