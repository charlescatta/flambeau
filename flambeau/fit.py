import torch
from torch import nn, optim
from typing import Callable, List
from .callbacks import TrainingContext, Stages
from .callbacks.internal import internal_train_loop_callbacks

def fit(n_epochs: int,
       model: nn.Module,
       optimizer: optim.Optimizer,
       loss_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
       data: torch.utils.data.DataLoader,
       callbacks: List[Callable] = []):
  """ Callback enabled training loop """
  ctx = TrainingContext(n_epochs, model, optimizer, loss_func, data, [*internal_train_loop_callbacks, *callbacks])
  ctx.at_stage(Stages.start_fit)
  for epoch in range(ctx.n_epochs):
    ctx.at_stage(Stages.start_epoch)
    ctx.epoch = epoch
    for X_b, y_b in ctx.data:
      ctx.X_b = X_b
      ctx.y_b = y_b
      ctx.at_stage(Stages.start_batch)
      ctx.at_stage(Stages.after_pred)
      ctx.at_stage(Stages.after_loss)
      ctx.at_stage(Stages.after_backward)
      ctx.at_stage(Stages.after_step)
      ctx.at_stage(Stages.end_batch)
    ctx.at_stage(Stages.end_epoch)
  ctx.at_stage(Stages.end_fit)