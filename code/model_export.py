import os
import sys
import glob
import argparse
import numpy as np
import PIL.Image as pil
import tensorrt as trt

import torch
from torchvision import transforms, datasets
import ctypes

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

import utility


def generate_trt_engine():

  import pandas as pd
  print(f'[trace] exec@generate_trt_engine')
  data_dir = '../data/chest_xray'
  _train_trans = utility.data_transforms(utility.TRAIN)
  trainset = datasets.ImageFolder(os.path.join(data_dir, utility.TRAIN), transform=_train_trans)
  default_batch_size = 16
  train_loader = torch.utils.data.DataLoader(
    trainset,
    batch_size=default_batch_size,
    pin_memory=False,
    drop_last=False,
    shuffle=True,
    num_workers=2,
    # sampler=BalanceClassSampler(labels=train_['label'].values, mode="downsampling")
  )

  images, labels = iter(train_loader).next()
  cache_file = "calibration.cache"
  element_bytes = images.shape[1] * images.shape[2] * images.shape[3] * 4

  calib = utility.Calibrator(train_loader, cache_file, element_bytes)
  # engine = build_engine_from_onnxmodel_int8(onnxmodel, calib)

  onnx_file_path = '../models/classify-sim.onnx'
  print(f'[trace] convert onnx file {onnx_file_path} to TensorRT engine')
  if not os.path.exists(onnx_file_path):
    print(f'[trace] target file {onnx_file_path} not exist, exiting')
    exit(-1)

  batch_size = 16
  mode: utility.CalibratorMode = utility.CalibratorMode.INT8
  TRT_LOGGER = trt.Logger()
  EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

  with trt.Builder(TRT_LOGGER) as builder, \
      builder.create_network(EXPLICIT_BATCH) as network, \
      builder.create_builder_config() as config, \
      trt.OnnxParser(network, TRT_LOGGER) as parser, \
      trt.Runtime(TRT_LOGGER) as runtime:

    # Parse model file
    print("[trace] loading ONNX file from path {}...".format(onnx_file_path))
    with open(onnx_file_path, "rb") as model:
      print("[trace] beginning ONNX file parsing")
      if not parser.parse(model.read()):
        print("[error] failed to parse the ONNX file.")
        for error in range(parser.num_errors):
          print(parser.get_error(error))
        return None
      print("[trace] completed parsing of ONNX file")

    builder.max_batch_size = batch_size
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, utility.GiB(4))

    if mode == utility.CalibratorMode.INT8:
      config.set_flag(trt.BuilderFlag.INT8)
    elif mode == utility.CalibratorMode.FP16:
      config.set_flag(trt.BuilderFlag.FP16)
    elif mode == utility.CalibratorMode.TF32:
      config.set_flag(trt.BuilderFlag.TF32)
    elif mode == utility.CalibratorMode.FP32:
      # do nothing since this is the default branch
      # config.set_flag(trt.BuilderFlag.FP32)
      pass
    else:
      print(f'[trace] unknown calibrator mode: {mode.name}, exit')
      exit(-1)

    config.int8_calibrator = calib
    engine_file_path = f'../models/efficientnet_b4_ns.{mode.name}.engine'
    input_channel = 3
    input_image_width = 224
    input_image_height = input_image_width
    network.get_input(0).shape = [batch_size, input_channel, input_image_width, input_image_height]
    return utility.build_engine_common_routine(network, builder, config, runtime, engine_file_path)
  pass

def export_to_onnx():
  output_model_name = '../models/classify.onnx'
  if os.path.exists(output_model_name):
    print(f'[trace] target onnx file {output_model_name} exist, return')
    return

  print(f'[trace] start to export to onnx model file:')
  full_pth_path = '../models/model_18th_epoch_8.770064184442163_loss.pth'
  print(f"[trace] Loading model from {full_pth_path}")

  if not os.path.exists(full_pth_path):
    print(f'[trace] target file {full_pth_path} not exist, exit...')
    return

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = utility.classify()
  model.load_state_dict(torch.load(full_pth_path))
  model.eval()

  model = model.to(device)
  print(f'[trace] model weight file loaded')

  input0 = torch.FloatTensor(1, 3, 224, 224).to(device)

  print(f'[trace] start the test run')
  example_output = model(input0)

  print('[trace] start to export the onnx file')
  print(f'[trace] current torch version: {torch.__version__}')

  _shape = (1, 3, 224, 224)
  utility.export2onnx(model, output_model_name, _shape)
  print(f"[trace] done with export_to_onnx")
  pass


def _main():
  print(f'[trace] working in the main function')
  export_to_onnx()
  generate_trt_engine()
  pass


if __name__ == '__main__':
  _main()
