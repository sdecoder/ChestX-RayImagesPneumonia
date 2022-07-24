'''
https://docs.nvidia.com/deeplearning/tensorrt/pytorch-quantization-toolkit/docs/tutorials/quant_resnet50.html

'''
import datetime
import os
import sys
import time
import collections

import torch
import torch.utils.data
from torch import nn

from tqdm import tqdm

import utils
import torchvision
from torchvision import transforms

from pytorch_quantization import nn as quant_nn
from pytorch_quantization import calib
from pytorch_quantization.tensor_quant import QuantDescriptor

from absl import logging
logging.set_verbosity(logging.FATAL)  # Disable logging as they are too noisy in notebook

# For simplicity, import train and eval functions from the train script from torchvision instead of copything them here
# Download torchvision from https://github.com/pytorch/vision
#sys.path.append("/raid/skyw/models/torchvision/references/classification/")
#from train import evaluate, train_one_epoch, load_data


quant_desc_input = QuantDescriptor(calib_method='histogram')
quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)

from pytorch_quantization import quant_modules
quant_modules.initialize()

model = torchvision.models.resnet50(pretrained=True, progress=False)
model.cuda()

data_path = "/raid/data/imagenet/imagenet_pytorch"
batch_size = 512

traindir = os.path.join(data_path, 'train')
valdir = os.path.join(data_path, 'val')
_args = collections.namedtuple('mock_args', ['model', 'distributed', 'cache_dataset'])

from torchsummary import summary


def collect_stats(model, num_batches):
  """Feed data to the network and collect statistic"""

  # Enable calibrators
  for name, module in model.named_modules():
    if isinstance(module, quant_nn.TensorQuantizer):
      if module._calibrator is not None:
        module.disable_quant()
        module.enable_calib()
      else:
        module.disable()

  summary(model,  (3, 224, 224))
  for i, (image, _) in tqdm(enumerate(data_loader), total=num_batches):
    model(image.cuda())
    if i >= num_batches:
      break

  # Disable calibrators
  for name, module in model.named_modules():
    if isinstance(module, quant_nn.TensorQuantizer):
      if module._calibrator is not None:
        module.enable_quant()
        module.disable_calib()
      else:
        module.enable()


def compute_amax(model, **kwargs):
  # Load calib result
  for name, module in model.named_modules():
    if isinstance(module, quant_nn.TensorQuantizer):
      if module._calibrator is not None:
        if isinstance(module._calibrator, calib.MaxCalibrator):
          module.load_calib_amax()
        else:
          module.load_calib_amax(**kwargs)
  #             print(F"{name:40}: {module}")
  model.cuda()

'''
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size,
    sampler=train_sampler, num_workers=4, pin_memory=True)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=batch_size,
    sampler=test_sampler, num_workers=4, pin_memory=True)
dataset, dataset_test, train_sampler, test_sampler = utils.load_data(traindir, valdir, _args(model='resnet50', distributed=False, cache_dataset=False))

'''


# It is a bit slow since we collect histograms on CPU
with torch.no_grad():
    collect_stats(model, num_batches=2)
    compute_amax(model, method="percentile", percentile=99.99)

criterion = nn.CrossEntropyLoss()
with torch.no_grad():
  utils.evaluate(model, criterion, data_loader_test, device="cuda", print_freq=20)

# Save the model
torch.save(model.state_dict(), "/tmp/quant_resnet50-calibrated.pth")
