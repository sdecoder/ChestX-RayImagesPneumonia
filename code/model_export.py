import os
import sys
import glob
import argparse
import numpy as np
import PIL.Image as pil
import tensorrt as trt
import nn_structure

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




def parse_args():
  parser = argparse.ArgumentParser(
    description='Simple testing function for Monodepthv2 models.')

  parser.add_argument('--name', default='raft', help="name your experiment")
  parser.add_argument('--stage', help="determines which dataset to use for training")
  parser.add_argument('--restore_ckpt', help="restore checkpoint")
  parser.add_argument('--small', action='store_true', help='use small model')
  parser.add_argument('--validation', type=str, nargs='+')

  parser.add_argument('--lr', type=float, default=0.00002)
  parser.add_argument('--num_steps', type=int, default=100000)
  parser.add_argument('--batch_size', type=int, default=6)
  parser.add_argument('--image_size', type=int, nargs='+', default=[384, 512])
  # parser.add_argument('--gpus', type=int, nargs='+', default=[0,1])
  parser.add_argument('--gpus', type=int, nargs='+', default=[0])
  parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

  parser.add_argument('--iters', type=int, default=12)
  parser.add_argument('--wdecay', type=float, default=.00005)
  parser.add_argument('--epsilon', type=float, default=1e-8)
  parser.add_argument('--clip', type=float, default=1.0)
  parser.add_argument('--dropout', type=float, default=0.0)
  parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')
  parser.add_argument('--add_noise', action='store_true')

  return parser.parse_args()

def export_to_onnx(args):

  print(f'[trace] start to export to onnx model file:')
  full_pth_path = './model_18th_epoch_8.770064184442163_loss.pth'
  print(f"[trace] Loading model from {full_pth_path}")

  if not os.path.exists(full_pth_path):
    print(f'[trace] target file {full_pth_path} not exist, exit...')
    return

  device = torch.device("cuda")
  model = nn_structure.classify()
  model.load_state_dict(torch.load(full_pth_path))
  model.eval()

  model = model.to(device)
  print(f'[trace] model weight file loaded')

  input0 = torch.FloatTensor(1, 3, 224, 224).to(device)
  output_model_name = 'classify.onnx'
  print(f'[trace] start the test run')
  example_output = model(input0)

  print('[trace] start to export the onnx file')
  print(f'[trace] current torch version: {torch.__version__}')

  dynamic_axes = {
    'input': {0: 'batch_size'}
  }
  torch.onnx.export(model,  # model being run
                    args=input0,  # model input (or a tuple for multiple inputs)
                    f=output_model_name,
                    export_params=True,
                    opset_version=16,
                    do_constant_folding=True,
                    input_names=['input', ],  # the model's input names
                    output_names=['output', ],
                    dynamic_axes=dynamic_axes)
  pass


def main(args):
  print(f'[trace] working in the main function')
  export_to_onnx(args)
  pass


if __name__ == '__main__':
  args = parse_args()
  main(args)
