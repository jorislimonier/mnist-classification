import torch
import torchvision
from torchvision.datasets.mnist import MNIST


class DataOperations:
  def __init__(self) -> None:
    self._load_data()
    self._normalize_data()

  def _load_data(self):
    self.train = MNIST(
      root="./data/",
      train=True,
      download=True,
      transform=torchvision.transforms.ToTensor(),
    )
    self.test = MNIST(
      root="./data/",
      train=False,
      download=True,
      transform=torchvision.transforms.ToTensor(),
    )
    self.X_train = self.train.data
    self.y_train = self.train.targets
    self.X_test = self.test.data
    self.y_test = self.test.targets

  def _normalize_data(self):
    self.X_train = self.X_train / self.X_train.max()
    self.X_test = self.X_test / self.X_test.max()


class Layer:
  def __init__(self):
    self.inputs = None
    self.outputs = None

  def forward_propagation(self, inputs):
    raise NotImplementedError

  def backward_propagation(self, outputs_error, lr):
    raise NotImplementedError


class FCLayer(Layer):
  def __init__(self, in_features: int, out_features: int) -> None:
    self.weights = torch.rand(size=(in_features, out_features)) - 0.5
    self.biases = torch.rand(size=(1, out_features)) - 0.5

  def forward(self, inputs: torch.Tensor):
    self.inputs = inputs
    self.outputs = torch.matmul(self.inputs, self.weights) + self.bias
    return self.outputs

  def back(self):
    raise NotImplementedError


class ActivationLayer(Layer):
  def __init__(self):
    pass


class Predict:
  def __init__(self) -> None:
    self.do = DataOperations()
    self.do.X_train = self.do.X_train.reshape(-1, 28 * 28)
    self.do.X_test = self.do.X_test.reshape(-1, 28 * 28)


class Network:
  def __init__(self) -> None:
    self.layers = []
    self.loss = None
    self.loss_deriv = None

  def add_layer(self, layer):
    self.layers.append(layer)