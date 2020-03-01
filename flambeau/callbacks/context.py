import torch
import warnings
from torch import optim, nn
from enum import Enum
from typing import Callable, Union, List, Iterable, Tuple

Stages = Enum('Stages', ' '.join([
  'start_fit',
  'start_epoch',
  'start_batch',
  'after_pred',
  'after_loss',
  'after_backward',
  'after_step',
  'end_batch',
  'end_epoch',
  'end_fit',
  'all_stages'
]))

class TrainingContext():
  globally_registered_callbacks = []
  def __init__(self,
               n_epochs: int,
               model: nn.Module,
               optimizer: optim.Optimizer,
               loss_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
               data: Union[torch.utils.data.DataLoader, Iterable[Tuple[torch.Tensor, torch.Tensor]]],
               callbacks: List[Callable]):
    self.model = model
    self.optimizer = optimizer
    self.optimizer.zero_grad()
    self.data = data
    self.loss_func = loss_func
    self.iters = 0
    self.n_epochs = n_epochs
    self.epoch = 0
    self.X_b = None
    self.y_b = None
    self.loss = None
    self.preds = None
    self.stage = None
    self.callbacks = self._create_callbacks_dict([*self.globally_registered_callbacks, *callbacks])
  
  @staticmethod
  def _create_callbacks_dict(callbacks: List[Callable]):
    out = { s.value: [] for s in Stages if s != Stages.all_stages }
    # callbacks = callbacks.sort(key=lambda c: c._order)
    for callback in callbacks:
      has_stage = hasattr(callback, '_stage')
      if not has_stage:
        callback._stage = Stages.all_stages
        callback._is_internal = False
      if has_stage and callback._stage != Stages.all_stages:
        out[callback._stage.value].append(callback)
      else:
        for stage in out.keys():
          out[stage].append(callback)
    
    return out

  def at_stage(self, stage: Stages):
    self.stage = stage
    for callback in self.callbacks[stage.value]:
      callback(self)

def train_callback(stage: Stages, order: int = 0, auto: bool=False):
  """ Decorator that turns a function into a valid callback """
  def wrapper(func: Callable):
    func._stage = stage
    func._order = order
    func._is_internal = False
    if auto:
      TrainingContext.globally_registered_callbacks.append(func)
    return func
  return wrapper