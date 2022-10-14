import torch
import torchvision
from torchvision import transforms

print("lalala")
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
    transform=transforms.Compose(
      transforms=[transforms.ToTensor(), transforms.Normalize((0,), (1,))]
    ),
  )
  test = torchvision.datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    # transform=ToTensor(),
  )

  # Extract tensor of data and labels for both the training and the test set
  X_train, y_train = train.data, train.targets
  X_test, y_test = test.data.float(), test.targets

  if normalize:
    X_train /= 255
    X_test /= 255

  return X_train, y_train, X_test, y_test
