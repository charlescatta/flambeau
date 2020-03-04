import torch
import warnings
from torch import optim, nn
from enum import Enum, auto
from typing import Callable, Union, List, Iterable, Tuple, Optional

class Stage(Enum):
  start_fit = auto()
  start_epoch = auto()
  start_batch = auto()
  after_pred = auto()
  after_loss = auto()
  after_backward = auto()
  after_step = auto()
  end_batch = auto()
  end_epoch = auto()
  end_fit = auto()
stage_names = [s.name for s in Stage]
Stage.all: Tuple[Stage] = tuple(s for s in Stage)

class CallbackFunc(Callable):
  def __init__(self,
               func: Callable,
               stage: Union[Stage, Iterable[Stage]] = Stage.all,
               order: int = 0,
               auto: bool = False,
               _is_internal: bool = False):
    if isinstance(func, CallbackFunc):
      return
    self.func = func
    if isinstance(stage, Iterable): 
      self.stage = stage
    else:
      self.stage = [stage]
    self._is_internal = _is_internal
    self._order = order
    if auto:
      TrainingContext.globally_registered_callbacks.append(self)

  def __new__(cls, func: Callable, *args, **kwargs):
    if isinstance(func, CallbackFunc):
      return func
    return super(CallbackFunc, cls).__new__(cls, *args, **kwargs)
  
  def __call__(self, *args, **kwargs):
    return self.func(*args, **kwargs)
  
  def __repr__(self):
    return f"{self.func.__name__}{': ' + self.func.__doc__ if self.func.__doc__  else ''}"

class CallbacksDict(dict):
  def __init__(self, callbacks: Iterable[Callable] = []):
    super().__init__()
    for s in Stage:
      self[s] = []
    self.register(callbacks)

  def register(self, callbacks: Iterable[Callable]):
    for c in callbacks:
      if isinstance(c, Callback):
        self.merge(c.to_dict())
      else:
        c = CallbackFunc(c)
        for stage in c.stage:
          self[stage].append(c)
          # TODO: implement sorted insert
    self.sort()
  
  def sort(self):
    for s in Stage:
      self[s] = list(sorted(self[s], key=lambda c: c._order))

  def merge(self, other):
    for stage in Stage:
      self[stage] = [*self[stage], *other[stage]]
    self.sort()
    return self
  
  def __repr__(self):
    out = ""
    for s in Stage:
      out += f"{s.name}\n"
      for f in self[s]:
        out += f"  |  {f} \n"
    return out

class Callback:
  _order = 0
  _is_internal=False
  @staticmethod
  def method_to_cbfunc(instance, name):
    def wrapper(*args, **kwargs):
      return getattr(instance, name)(*args, **kwargs)
    
    wrapper.__name__ = type(instance).__name__
    wrapper.__doc__ = getattr(instance, name).__doc__ or type(instance).__doc__ or None
    
    return CallbackFunc(wrapper, Stage[name], order=instance._order, _is_internal=instance._is_internal)
  
  def to_dict(self) -> CallbacksDict:
    methods = [ method for method in dir(self) if method in stage_names ]
    return CallbacksDict([ Callback.method_to_cbfunc(self, method) for method in methods ])

class TrainingContext():
  globally_registered_callbacks = []
  def __init__(self,
               n_epochs: int,
               model: nn.Module,
               optimizer: optim.Optimizer,
               loss_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
               data: Union[torch.utils.data.DataLoader, Iterable[Tuple[torch.Tensor, torch.Tensor]]],
               callbacks: Iterable[Callable] = []):
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
    self.stage = Stage.start_fit
    self.callbacks = CallbacksDict([*TrainingContext.globally_registered_callbacks, *callbacks])

  def at_stage(self, stage: Stage):
    self.stage = stage
    for callback in self.callbacks[stage]:
      callback(self)

def train_callback(stage: Union[Stage, Iterable[Stage]], order: int = 0, auto: bool=False):
  """ Decorator that turns a function into a valid callback """
  def wrapper(func: Callable):
    return CallbackFunc(func, stage, order, auto)
  return wrapper