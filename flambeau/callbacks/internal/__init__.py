from typing import Callable
from ..context import TrainingContext, Stages, train_callback

INTERNAL_TRAIN_ORDER = -9999
internal_train_loop_callbacks = []

def internal_train_callback(stage: Stages, order: int = 0):
  def wrapper(func: Callable):
    func = train_callback(stage, INTERNAL_TRAIN_ORDER)(func)
    func._is_internal = True
    internal_train_loop_callbacks.append(func)
    return func
  return wrapper

@internal_train_callback(Stages.start_batch)
def set_model_to_training_mode(ctx: TrainingContext):
  """ Sets the model to training mode at each batch start """
  ctx.model.train()

@internal_train_callback(Stages.after_pred)
def compute_preds(ctx: TrainingContext):
  """ Compute model predictions """
  ctx.preds = ctx.model(ctx.X_b)

@internal_train_callback(Stages.after_loss)
def compute_loss(ctx: TrainingContext):
  """ Compute the loss with the given loss function """
  ctx.loss = ctx.loss_func(ctx.preds, ctx.y_b)

@internal_train_callback(Stages.after_backward)
def compute_loss_grad(ctx: TrainingContext):
  """ Compute parameter gradients with respect to loss """
  ctx.loss.backward()

@internal_train_callback(Stages.after_step)
def compute_step(ctx: TrainingContext):
  """ Backpropagation """
  ctx.optimizer.step()

@internal_train_callback(Stages.after_step)
def clear_optim_grad(ctx: TrainingContext):
  """ Clear the optimizer's gradients """
  ctx.optimizer.zero_grad()

@internal_train_callback(Stages.end_batch)
def set_model_eval(ctx: TrainingContext):
  """ Sets the model to evaluation mode at each batch end """
  ctx.model.eval()

@internal_train_callback(Stages.end_batch)
def increment_iters(ctx: TrainingContext):
  """ Increment iteration count on minibatch end """
  ctx.iters += 1