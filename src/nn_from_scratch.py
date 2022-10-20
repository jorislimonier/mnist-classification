# %%
from typing import Callable
import numpy as np
import pandas as pd
import plotly.express as px
from keras.datasets import mnist
from sklearn.preprocessing import OneHotEncoder

# %%
def visualize_digits(n_digits: int = 12):
  fig = px.imshow(X_train[:10, :, :], binary_string=True, animation_frame=0)
  # fig = px.imshow(X_train[:10, :, :], binary_string=True, facet_col=0, facet_col_wrap=5)
  return fig


# %%
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(60000, -1)
X_test = X_test.reshape(10000, -1)
# X_train = pd.DataFrame(X_train)
# X_test = pd.DataFrame(X_test)


#%%
np.random.seed(42)


class Layer:
  def __init__(self):
    self.input = None
    self.output = None

  def forward(self, input_data):
    raise NotImplementedError

  def backward(self, output_error, lr):
    raise NotImplementedError


class FCLayer(Layer):
  def __init__(self, input_size: int, output_size: int):
    self.weights = np.random.rand(input_size, output_size) - 0.5
    self.bias = np.random.rand(1, output_size) - 0.5

  def forward(self, input_data):
    """Forward pass"""
    self.input = input_data
    self.output = np.dot(self.input, self.weights) + self.bias
    return self.output

  def backward(self, output_error: np.ndarray, lr: float):
    input_error = np.dot(output_error, self.weights.T)
    weights_error = np.dot(self.input.T, output_error)

    # update parameters
    self.weights -= lr * weights_error
    self.bias -= lr * output_error
    return input_error


class ActivationLayer(Layer):
  def __init__(self, activation: Callable, activation_deriv: Callable):
    self.activation = activation
    self.activation_deriv = activation_deriv

  def forward(self, input_data):
    self.input = input_data
    self.output = self.activation(self.input)
    return self.output

  def backward(self, output_error, lr):
    return self.activation_deriv(self.input) * output_error


batch_size = 1

X_batch = X_train[:batch_size]
y_batch = y_train[:batch_size]

y_batch_ohe = np.zeros((len(y_batch), 10), dtype=int)
for idx, ohe in enumerate(y_batch_ohe):
  ohe[y_batch[idx]] = 1

fcl = FCLayer(input_size=784, output_size=10)
epochs = 100
for epoch in range(epochs):
  out = fcl.forward(X_batch)
  err = (out - y_batch_ohe) ** 2

  fcl.backward(err, lr=0.3)
  print(fcl.output)
# %%

y_batch_ohe
