import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import torch
import numpy as np
import torch.nn.functional as F

from torch.autograd import Variable


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)  # hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output)  # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))  # activation function for hidden layer
        x = self.predict(x)  # linear output


net = Net(4991, 10, 1)
print(net)
optimizer = torch.optim.SGD(net.parameters(), lr=0.2)
loss_func = torch.nn.MSELoss()
plt.ion()  # something about plotting


def read_train_file():
    df = pd.read_csv("train.csv")
    print("Shape( " + str(df.shape))
    del df["ID"]
    print("Shape( " + str(df.shape))
    return df


def train(x, y):
    for t in range(200):
        prediction = net(X_train)  # input x and predict based on x

    loss = loss_func(prediction, y)  # must be (1. nn output, 2. target)

    optimizer.zero_grad()  # clear gradients for next train
    loss.backward()  # backpropagation, compute gradients
    optimizer.step()  # apply gradients

    if t % 5 == 0:
        # plot and show learning process
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    df = read_train_file()
    X_train, X_test = train_test_split(df, test_size=0.2)
    print(X_train.shape)
    print(X_test.shape)
    y_train = X_train["target"]
    y_test = X_test["target"]
    del X_train["target"]
    del X_test["target"]

    X_train = np.array(X_train.values)
    y_train = np.array(torch.Tensor(y_train.values))
    x = train(Variable(torch.from_numpy(X_train)), Variable(torch.from_numpy(y_train)))