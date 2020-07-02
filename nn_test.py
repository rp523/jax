#coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from matplotlib import pyplot as plt
import os, time, argparse

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        mid = 300
        self.__layers = []
        self.__layers.append(nn.Linear(2, mid))
        self.__layers.append(nn.ReLU)
        self.__layers.append(nn.Linear(mid, mid))
        self.__layers.append(nn.ReLU)
        self.__layers.append(nn.Linear(mid, 1))

    def forward(self, x):
        for layer in self.__layers:
            x = layer(x)
        return x

def main():
    net = Net()
    BATCH_SIZE = 64
    #sampler = Sampler(BATCH_SIZE)   
    criterion = nn.MSELoss()
    optimizer = optim.adam(net.parameters(), lr=0.001, betas = (0.5, 0.9))
    for epoch in range(10000):
        running_loss = 0.0

        x = np.uniform((BATCH_SIZE, 2)) - 0.5
        #from ebm.sampler import Sampler
        #y = Sampler.prob(x)
        y = np.uniform((BATCH_SIZE, 2)) - 0.5

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        pred = net(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        print(epoch, running_loss)
    bin_num = 100
    delta = 0.5
    x = np.linspace(-delta, delta, bin_num)
    x = np.tile(x.reshape(1, -1), (bin_num, 1))
    y = np.linspace(-delta, delta, bin_num)
    y = np.tile(y.reshape(-1, 1), (1, bin_num))
    data = np.append(x.reshape(-1, 1), y.reshape(-1, 1), axis = 1)
    assert(data.shape == (bin_num * bin_num, 2))
    minus_E = net(data)
    assert(minus_E.shape == (bin_num * bin_num, 1))
    unnorm_log_q = minus_E
    unnorm_log_q = unnorm_log_q.reshape((bin_num, bin_num))
    X = np.linspace(-delta, delta, bin_num)
    Y = np.linspace(-delta, delta, bin_num)
    X, Y = np.meshgrid(X, Y)
    plt.pcolor(X, Y, unnorm_log_q)
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    main()
    print("Done.")