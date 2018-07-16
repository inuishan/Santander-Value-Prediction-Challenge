import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torch import nn, optim


import torch
import numpy as np

from torch.autograd import Variable



# Linear Regression Model
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1)  # input and output is 1 dimension

    def forward(self, x):
        out = self.linear(x)
        return out

model = LinearRegression()
# 定义loss和优化函数
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=1e-4)

# 开始训练
num_epochs = 1000


def read_train_file():
    df = pd.read_csv("train.csv")
    print("Shape( " + str(df.shape))
    del df["ID"]
    print("Shape( " + str(df.shape))
    return df




if __name__ == "__main__":
    df = read_train_file()
    X_train, X_test = train_test_split(df, test_size=0.2)
    print(X_train.shape)
    print(X_test.shape)
    y_train = X_train["target"]
    y_test = X_test["target"]
    del X_train["target"]
    del X_test["target"]

    X_array = np.array(X_train.values)
    X_train = torch.from_numpy(X_array).float()
    y_array = np.array(y_train.values)

    y_arr_mod = []

    for val in y_array:
        y_arr_mod.append([val])
    y_array = y_arr_mod
    y_final_array = np.array(y_array)
    y_train = torch.from_numpy(y_final_array).float()
    for epoch in range(num_epochs):
        inputs = Variable(X_train)
        target = Variable(y_train)

        # forward
        out = model(inputs)
        loss = criterion(out, target)
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch+1) % 20 == 0:
            print('Epoch[{}/{}], loss: {:.6f}'
                  .format(epoch+1, num_epochs, loss.data[0]))

    model.eval()
    predict = model(Variable(X_train))
    predict = predict.data.numpy()
    plt.plot(X_train.numpy(), y_train.numpy(), 'ro', label='Original data')
    plt.plot(X_train.numpy(), predict, label='Fitting Line')
    # 显示图例
    plt.legend()
    plt.show()
    torch.save(model.state_dict(), './linear.pth')