import torch
import torchvision
import flambeau
from torchvision import transforms, models
from torch import nn, optim, functional
from flambeau.callbacks import train_callback, TrainingContext, Stage, Callback
from flambeau.callbacks.gpu import to_gpu_callbacks

N_EPOCHS = 2
BATCH_SIZE = 128
LR=3e-3

tfms = transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize(torch.tensor([0.485, 0.456, 0.406]), #Imagenet stats, TODO: change to CIFAR10 stats
                        torch.tensor([0.229, 0.224, 0.225]))
])

train_ds = torchvision.datasets.CIFAR10('./test_data', 'train', download=True, transform=tfms)
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

def get_model(n_classes: int):
  m = models.resnet18(pretrained=True)
  in_features = m.fc.in_features
  m.fc = nn.Linear(in_features, n_classes)
  return m

n_classes = next(iter(train_dl))[-1].shape[-1]
m = get_model(n_classes)

optimizer = optim.AdamW(m.parameters(), lr=LR)
loss_func = nn.functional.cross_entropy

@train_callback(Stage.start_epoch, auto=True)
def print_epochs(ctx: TrainingContext):
  """ Prints current epoch number at epoch start """
  print(f"Epoch {ctx.epoch + 1}/{ctx.n_epochs}")

@train_callback(Stage.end_batch, auto=True)
def print_iters(ctx: TrainingContext):
  """ Print iterations after batch end """
  print(f"Number of batches processed: {ctx.iters} | Loss: {ctx.loss.item():.3f}")

def some_normal_func(ctx):
  """ This will appear in the callbacks summary """
  if ctx.stage == Stage.start_fit:
    print("QUICK GET THE CAMERA, IT'S WORKING")
  if ctx.iters < 3:
    print(f"Hello from {ctx.stage.name}")
  if ctx.iters == 3 and ctx.stage == Stage.start_batch:
    print("Ok gonna stop spamming now")

class ValidatorCallback(Callback):
  def __init__(self):
    self.tracked_epochs = 0
  
  def start_fit(self):
    print(f'Hello {self.tracked_epochs}')

# TODO: Write validation pass example callback


if torch.cuda.device_count() > 0:
  print("CUDA GPU found, will use one gpu for training")
  callbacks = [*callbacks, *to_gpu_callbacks]
else:
  print("No GPU found, will train on CPU")

class Recorder(Callback):
  """ Metrics recording class """
  def __init__(self):
    self.internal_state = 42
  
  def after_backward(self, ctx):
    """ print data after backwards """
    print(self.internal_state, '\n\n\n')
    

callbacks = [some_normal_func, Recorder()]

# Print a summary of the callbacks
flambeau.callbacks.summarize(callbacks, show_internal=True)
# Start training
flambeau.fit(N_EPOCHS, m, optimizer, loss_func, train_dl, callbacks=callbacks)