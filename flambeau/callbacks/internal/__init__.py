from typing import Callable
from ..context import TrainingContext, Stage, CallbackFunc

INTERNAL_TRAIN_ORDER = -9999

def internal_train_callback(stage: Stage, order: int = INTERNAL_TRAIN_ORDER):
  def wrapper(func: Callable):
    return CallbackFunc(func, stage, order, _is_internal=True, auto=True)
  return wrapper

@internal_train_callback(Stage.start_batch)
def set_model_to_training_mode(ctx: TrainingContext):
  """ Sets the model to training mode at each batch start """
  ctx.model.train()

@internal_train_callback(Stage.after_pred)
def compute_preds(ctx: TrainingContext):
  """ Compute model predictions """
  ctx.preds = ctx.model(ctx.X_b)

@internal_train_callback(Stage.after_loss)
def compute_loss(ctx: TrainingContext):
  """ Compute the loss with the given loss function """
  ctx.loss = ctx.loss_func(ctx.preds, ctx.y_b)

@internal_train_callback(Stage.after_backward)
def compute_loss_grad(ctx: TrainingContext):
  """ Compute parameter gradients with respect to loss """
  ctx.loss.backward()

@internal_train_callback(Stage.after_step)
def compute_step(ctx: TrainingContext):
  """ optimizer backpropagation """
  ctx.optimizer.step()

@internal_train_callback(Stage.after_step)
def clear_optim_grad(ctx: TrainingContext):
  """ Clear the optimizer's gradients """
  ctx.optimizer.zero_grad()

@internal_train_callback(Stage.end_batch)
def set_model_eval(ctx: TrainingContext):
  """ Sets the model to evaluation mode at each batch end """
  ctx.model.eval()

@internal_train_callback(Stage.end_batch)
def increment_iters(ctx: TrainingContext):
  """ Increment iteration count on minibatch end """
  ctx.iters += 1