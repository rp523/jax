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
        self.l1 = nn.Linear(2, mid)
        self.l2 = nn.Linear(mid, mid)
        self.l3 = nn.Linear(mid, 1)

    def forward(self, x):
        x = self.l1(x)
        x = F.relu(x)
        x = self.l2(x)
        x = F.sigmoid(x)
        x = self.l3(x)
        x = F.sigmoid(x)
        return x

def main(is_training):
    MODEL_PATH = "tmp.bin"

    net = Net().float()
    BATCH_SIZE = 64
    #sampler = Sampler(BATCH_SIZE)   
    criterion = nn.MSELoss(reduction = "sum")
    optimizer = optim.Adam(net.parameters(), lr=0.001, betas = (0.5, 0.9))
    t0 = time.time()
    save_t0 = t0
    epoch = 0
    while is_training:
        running_loss = 0.0

        x = np.random.uniform((BATCH_SIZE, 2)) - 0.5
        from ebm.sampler import Sampler
        y = Sampler.prob(x)
        x = torch.from_numpy(np.array(x)).float()
        y = torch.from_numpy(np.array(y)).float()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        pred = net(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()

        t1 = time.time()
        if t1 - t0 > 1:
            print(  epoch,
                    "{:.1f}ms".format((t1 - t0) * 1000),
                    loss.item(),
                    )
            t0 = t1

        save_t1 = time.time()
        if (save_t1 - save_t0 > 10):
            save_t0 = save_t1
            save_graph(net)

def save_graph(arg_net):
    bin_num = 100
    delta = 0.5
    x = np.linspace(-delta, delta, bin_num)
    x = np.tile(x.reshape(1, -1), (bin_num, 1))
    y = np.linspace(-delta, delta, bin_num)
    y = np.tile(y.reshape(-1, 1), (1, bin_num))
    data = np.append(x.reshape(-1, 1), y.reshape(-1, 1), axis = 1)
    data = torch.from_numpy(np.array(data)).float()
    assert(data.shape == (bin_num * bin_num, 2))
    minus_E = arg_net(data)
    assert(minus_E.shape == (bin_num * bin_num, 1))
    unnorm_log_q = minus_E
    unnorm_log_q = unnorm_log_q.reshape((bin_num, bin_num))
    X = np.linspace(-delta, delta, bin_num)
    Y = np.linspace(-delta, delta, bin_num)
    X, Y = np.meshgrid(X, Y)
    plt.clf()
    plt.pcolor(X, Y, unnorm_log_q.detach().numpy())
    plt.colorbar()
    plt.savefig("tmp.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--training", type = int)
    args = parser.parse_args()
    is_training = (args.training != 0)
    main(is_training)
    print("Done.")