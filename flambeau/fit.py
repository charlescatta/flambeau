import torch
from torch import nn, optim
from typing import Callable, List
from .callbacks import TrainingContext, Stage, internal

def fit(n_epochs: int,
       model: nn.Module,
       optimizer: optim.Optimizer,
       loss_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
       data: torch.utils.data.DataLoader,
       callbacks: List[Callable] = []):
  """ Callback enabled training loop """
  ctx = TrainingContext(n_epochs, model, optimizer, loss_func, data, callbacks)
  ctx.at_stage(Stage.start_fit)
  for epoch in range(ctx.n_epochs):
    ctx.at_stage(Stage.start_epoch)
    ctx.epoch = epoch
    for X_b, y_b in ctx.data:
      ctx.X_b = X_b
      ctx.y_b = y_b
      ctx.at_stage(Stage.start_batch)
      ctx.at_stage(Stage.after_pred)
      ctx.at_stage(Stage.after_loss)
      ctx.at_stage(Stage.after_backward)
      ctx.at_stage(Stage.after_step)
      ctx.at_stage(Stage.end_batch)
    ctx.at_stage(Stage.end_epoch)
  ctx.at_stage(Stage.end_fit)