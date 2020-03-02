from .context import TrainingContext, Stage, CallbacksDict
from typing import List, Callable

def summarize(callbacks: List[Callable], show_internal: bool = False):
  """
    Prints out a summary of the callbacks and their order
  """
  callbacks = [*TrainingContext.globally_registered_callbacks, *callbacks]
  # if show_internal:
  #   callbacks.append(*TrainingContext.g)
  callbacks_dict = CallbacksDict(callbacks)
  print(f"{'='*40} SUMMARY {'='*40}")
  print(f"Autoregistered callbacks: {len(TrainingContext.globally_registered_callbacks)}\n")
  print(callbacks_dict)

