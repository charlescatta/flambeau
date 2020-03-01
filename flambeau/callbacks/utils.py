from . import TrainingContext, Stages
from typing import List, Callable

def summarize(callbacks: List[Callable], show_internal: bool = False):
  """
    Prints out a summary of the callbacks and their order
  """
  callbacks = [*TrainingContext.globally_registered_callbacks, *callbacks]
  if show_internal:
    callbacks = [*internal_train_loop_callbacks, *callbacks]
  callbacks_dict = TrainingContext._create_callbacks_dict(callbacks)
  print(f"{'='*40} SUMMARY {'='*40}")
  print(f"Autoregistered callbacks: {len(TrainingContext.globally_registered_callbacks)}\n")
  for stage, cbs in callbacks_dict.items():
    if len(cbs):
      print(f"{'='*12} {stage}: {Stages(stage).name} {'='*12}")
      for cb in cbs:
        print(f"{'*' if cb._is_internal else ''} {cb.__name__}: {cb.__doc__ if cb.__doc__ else ''}")
      print('\n')
  if show_internal:
    print("* := internal callbacks")
