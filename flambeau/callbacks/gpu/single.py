"""
  Single GPU training callbacks
"""
import torch
from typing import Union
from ..context import TrainingContext, train_callback, Stages

def SendModelToDevice(device: Union[str, torch.device]):
  """ Create a callback that sends the model to the specified device at training startup """
  def func(ctx: TrainingContext):
    ctx.model = ctx.model.to(device=device)
  func.__doc__ = f"Send model to device {device}"
  func.__name__ = f"send_model_to_{device}" # Is this legal?
  return train_callback(Stages.start_fit)(func)

def SendBatchToDevice(device: Union[str, torch.device]):
  """ Create a callback that sends the current minibatch to the specified device at each minibatch start """
  def func(ctx: TrainingContext):
    ctx.X_b = ctx.X_b.to(device=device)
    ctx.y_b = ctx.y_b.to(device=device)
  func.__doc__ = f"Send batch to device {device}"
  func.__name__ = f"send_batch_to_{device}" # Is this legal?
  return train_callback(Stages.start_batch)(func)

cuda_model = SendModelToDevice(torch.device('cuda'))
cuda_batch = SendBatchToDevice(torch.device('cuda'))

to_gpu_callbacks = [cuda_model, cuda_batch]