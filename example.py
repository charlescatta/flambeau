import torch
import torchvision
import flambeau
from torchvision import transforms, models
from torch import nn, optim, functional
from flambeau.callbacks import train_callback, TrainingContext, Stages
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

# This callback will be automatically registered
# and does not need to be passed to the fit function
@train_callback(Stages.start_epoch, auto=True)
def print_epochs(ctx: TrainingContext):
  """ Prints current epoch number at epoch start """
  print(f"Epoch {ctx.epoch + 1}/{ctx.n_epochs}")


# This callback can be passed to the fit function
@train_callback(Stages.end_batch)
def print_iters(ctx: TrainingContext):
  """ Print iterations after batch end """
  print(f"Number of batches processed: {ctx.iters} | Loss: {ctx.loss.item():.3f}")

# Any function that receives one positional parameter can be
# registered as a callback, it will be called at every stage of the
# training loop
def some_normal_func(ctx):
  """ This will appear in the callbacks summary """
  if ctx.stage == Stages.start_fit:
    print("QUICK GET THE CAMERA, IT'S WORKING")
  if ctx.iters < 3:
    print(f"Hello from {ctx.stage.name}")
  if ctx.iters == 3 and ctx.stage == Stages.start_batch:
    print("Ok gonna stop spamming now")

# TODO: Write validation pass example callback

callbacks = [print_iters, some_normal_func]

if torch.cuda.device_count() > 0:
  print("CUDA GPU found, will use one gpu for training")
  callbacks = [*callbacks, *to_gpu_callbacks]
else:
  print("No GPU found, will train on CPU")

# Print a summary of the callbacks
flambeau.callbacks.summarize(callbacks)

# Start training
flambeau.fit(N_EPOCHS, m, optimizer, loss_func, train_dl, callbacks=callbacks)