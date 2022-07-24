import datetime
import os
import time
import warnings

import presets
from sampler import RASampler

import torch
import torch.utils.data
import torchvision
import utils
from torch import nn
from torch.utils.data.dataloader import default_collate
from torchvision.transforms.functional import InterpolationMode


def _get_cache_path(filepath):
    import hashlib

    h = hashlib.sha1(filepath.encode()).hexdigest()
    cache_path = os.path.join("~", ".torch", "vision", "datasets", "imagefolder", h[:10] + ".pt")
    cache_path = os.path.expanduser(cache_path)
    return cache_path


def load_data(traindir, valdir, args):
  # Data loading code
  print("Loading data")
  #val_resize_size, val_crop_size, train_crop_size = args.val_resize_size, args.val_crop_size, args.train_crop_size
  val_resize_size=256
  val_crop_size=224
  train_crop_size=224
  interpolation='bilinear'

  #interpolation = InterpolationMode(args.interpolation)

  print("Loading training data")
  st = time.time()
  cache_path = _get_cache_path(traindir)
  if args.cache_dataset and os.path.exists(cache_path):
    # Attention, as the transforms are also cached!
    print(f"Loading dataset_train from {cache_path}")
    dataset, _ = torch.load(cache_path)
  else:
    auto_augment_policy = getattr(args, "auto_augment", None)
    random_erase_prob = getattr(args, "random_erase", 0.0)
    dataset = torchvision.datasets.ImageFolder(
      traindir,
      presets.ClassificationPresetTrain(
        crop_size=train_crop_size,
        interpolation=interpolation,
        auto_augment_policy=auto_augment_policy,
        random_erase_prob=random_erase_prob,
      ),
    )
    if args.cache_dataset:
      print(f"Saving dataset_train to {cache_path}")
      utils.mkdir(os.path.dirname(cache_path))
      utils.save_on_master((dataset, traindir), cache_path)
  print("Took", time.time() - st)

  print("Loading validation data")
  cache_path = _get_cache_path(valdir)
  if args.cache_dataset and os.path.exists(cache_path):
    # Attention, as the transforms are also cached!
    print(f"Loading dataset_test from {cache_path}")
    dataset_test, _ = torch.load(cache_path)
  else:
    if args.weights and args.test_only:
      weights = torchvision.models.get_weight(args.weights)
      preprocessing = weights.transforms()
    else:
      preprocessing = presets.ClassificationPresetEval(
        crop_size=val_crop_size, resize_size=val_resize_size, interpolation=interpolation
      )

    dataset_test = torchvision.datasets.ImageFolder(
      valdir,
      preprocessing,
    )
    if args.cache_dataset:
      print(f"Saving dataset_test to {cache_path}")
      utils.mkdir(os.path.dirname(cache_path))
      utils.save_on_master((dataset_test, valdir), cache_path)

  print("Creating data loaders")
  if args.distributed:
    if hasattr(args, "ra_sampler") and args.ra_sampler:
      train_sampler = RASampler(dataset, shuffle=True, repetitions=args.ra_reps)
    else:
      train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False)
  else:
    train_sampler = torch.utils.data.RandomSampler(dataset)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

  return dataset, dataset_test, train_sampler, test_sampler

#=========================================================================================================================
def evaluate(model, criterion, data_loader, device, print_freq=100, log_suffix=""):
  model.eval()
  metric_logger = utils.MetricLogger(delimiter="  ")
  header = f"Test: {log_suffix}"

  num_processed_samples = 0
  with torch.inference_mode():
    for image, target in metric_logger.log_every(data_loader, print_freq, header):
      image = image.to(device, non_blocking=True)
      target = target.to(device, non_blocking=True)
      output = model(image)
      loss = criterion(output, target)

      acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
      # FIXME need to take into account that the datasets
      # could have been padded in distributed setup
      batch_size = image.shape[0]
      metric_logger.update(loss=loss.item())
      metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
      metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
      num_processed_samples += batch_size
  # gather the stats from all processes

  num_processed_samples = utils.reduce_across_processes(num_processed_samples)
  if (
      hasattr(data_loader.dataset, "__len__")
      and len(data_loader.dataset) != num_processed_samples
      and torch.distributed.get_rank() == 0
  ):
    # See FIXME above
    warnings.warn(
      f"It looks like the dataset has {len(data_loader.dataset)} samples, but {num_processed_samples} "
      "samples were used for the validation, which might bias the results. "
      "Try adjusting the batch size and / or the world size. "
      "Setting the world size to 1 is always a safe bet."
    )

  metric_logger.synchronize_between_processes()

  print(f"{header} Acc@1 {metric_logger.acc1.global_avg:.3f} Acc@5 {metric_logger.acc5.global_avg:.3f}")
  return metric_logger.acc1.global_avg
