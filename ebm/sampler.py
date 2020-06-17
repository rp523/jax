#coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm

class Sampler:
    def __init__(self):
        self.__maker = Sampler.__Maker()
    def sample(self):
        return next(self.__maker)
    def __Maker():
        while True:
            x = np.random.uniform()
            y = np.random.uniform()
            z = np.random.uniform()
            if z < Sampler.__prob(x, y):
                yield x, y
    def __prob(x, y):
        delta_r = 0.2
        top_num = 3
        sigma = 0.05
        
        rx = x - 0.5
        ry = y - 0.5
        r = (rx ** 2 + ry ** 2) ** 0.5

        ret = 0.0
        for i in range(top_num):
            ret += np.exp(- ((r - (i + 0.5) * delta_r) ** 2) / (2 * sigma ** 2))
        return ret

def main():
    SAMPLE_NUM = 999999
    s = Sampler()
    x_vec = np.zeros(SAMPLE_NUM, dtype = np.float32)
    y_vec = np.zeros(SAMPLE_NUM, dtype = np.float32)
    for i in tqdm(range(SAMPLE_NUM)):
        x, y = s.sample()
        x_vec[i] = x
        y_vec[i] = y
    
    fig = plt.figure()
    ax = fig.add_subplot(111)

    H = ax.hist2d(x_vec, y_vec, bins=40, cmap=cm.jet)
    ax.set_title('1st graph')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    fig.colorbar(H[3],ax=ax)
    plt.show()

if __name__ == "__main__":
    main()
    print("Done.")