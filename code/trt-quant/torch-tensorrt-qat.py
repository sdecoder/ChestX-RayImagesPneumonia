'''

import unittest
import torch_tensorrt as torchtrt
from torch_tensorrt.logging import *
import torch
import torch.nn as nn
from torch.nn import functional as F

import torchvision
import torchvision.transforms as transforms
import torch_tensorrt

'''


#https://github.com/pytorch/TensorRT/issues/1026

import tensorrt
print(tensorrt.__version__)
