import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch
import torch.optim as optim  # Where the optimization modules are
import torchvision  # To be able to access standard datasets more easily
from plotly.subplots import make_subplots
from torch import nn
from torchvision import transforms
import torch.nn.functional as F



def load_data_torch(
  normalize: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  """Use torchvision to conveniently load some datasets.
  Return X_train, y_train, X_test, y_test
  """
  train = torchvision.datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    # transform=transforms.Compose(
    #   transforms=[transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)) ]
    # ),
    transform=transforms.ToTensor(),
  )
  test = torchvision.datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    # transform=ToTensor(),
  )

  # Extract tensor of data and labels for both the training and the test set
  X_train, y_train = train.data.float(), train.targets
  X_test, y_test = test.data.float(), test.targets

  if normalize:
    X_train /= 255
    X_test /= 255

  return X_train, y_train, X_test, y_test


def display_digits(
  X_train: torch.Tensor, y_train: torch.Tensor, nb_subplots: int = 12, cols: int = 4
) -> go.Figure:
  # Compute the rows and columns arrangements
  nb_subplots = 12
  cols = 4

  if nb_subplots % cols:
    rem = 1
  else:
    rem = 0

  rows = nb_subplots // cols + rem
  fig = make_subplots(
    rows=rows,
    cols=cols,
    subplot_titles=[f"Label: {int(y_train[idx])}" for idx in range(nb_subplots)],
    shared_xaxes=True,
    shared_yaxes=True,
    horizontal_spacing=0.02,
    vertical_spacing=0.1,
  )

  for idx in range(nb_subplots):
    row = (idx // cols) + 1
    col = idx % cols + 1
    img = X_train[idx]
    img = img.flip([0])
    trace = px.imshow(img=img, color_continuous_scale="gray")
    fig.append_trace(trace=trace.data[0], row=row, col=col)

  fig.update_layout(coloraxis_showscale=False)
  fig.update_xaxes(showticklabels=False)
  fig.update_yaxes(showticklabels=False)

  return fig


class DigitsNet(nn.Module):
  def __init__(self):
    super().__init__()
    n_features = 28 * 28
    n_classes = 10
    self.lin = nn.Linear(n_features, n_classes)

  def forward(self, xb):
    return self.lin(xb)
