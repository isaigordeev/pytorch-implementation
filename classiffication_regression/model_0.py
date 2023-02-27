from matplotlib import pyplot as plt

import torch
from torch import nn

weight = 0.3
bias = 0.9

x = torch.arange(start=0, end=1, step=0.01)
y = weight*x+bias

split = int(0.8*len(x))

train_x, train_y = x[:split], y[:split]
test_x, test_y = x[split:], y[split:]

def plot(train_x = train_x,
                     train_y = train_y,
                     test_x = test_x,
                     test_y = test_y,
                     pred = None):
    plt.clf()
    plt.scatter(test_x, test_y, c='r')
    plt.scatter(train_x, train_y, c='g')

    if pred is not None:
        plt.scatter(test_x, pred)

    plt.savefig('models/model_00.pdf')

def plot_predictions(pred, train_x = train_x,
                     train_y = train_y,
                     test_x = test_x,
                     test_y = test_y,
                     ):
    plt.clf()
    plt.scatter(test_x, test_y, c='r')
    plt.scatter(train_x, train_y, c='g')
    plt.scatter(test_x, pred)
    plt.savefig('models/model_00_pred.pdf')

def plot_loss(epochs, loss_train, loss_test):
    plt.clf()
    plt.scatter(epochs, loss_test, label='test')
    plt.scatter(epochs, loss_train, label='train')
    plt.legend()
    plt.savefig('models/model_00_loss.pdf')
    


class LinearRegressionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weights = nn.Parameter(torch.rand(1, dtype=torch.float),
                                    requires_grad=True)
        self.bias = nn.Parameter(torch.rand(1, dtype=torch.float),
                                    requires_grad=True)

    def forward(self, x):
        return x*self.weights + self.bias


    