#coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm

class Sampler:
    def __init__(self, batch_size = 1):
        self.__maker = Sampler.__Maker(batch_size)
    def sample(self):
        return next(self.__maker)
    def __Maker(batch_size):
        xy = np.empty((batch_size, 2), dtype = np.float32)
        valid_num = 0
        while True:
            x = np.random.uniform()
            y = np.random.uniform()
            z = np.random.uniform()
            if z < Sampler.__prob(x, y):
                xy[valid_num] = np.array([x, y])
                valid_num += 1
            if valid_num == batch_size:
                yield xy
                valid_num = 0
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
    SAMPLE_NUM = 99999
    s = Sampler()
    x_vec = np.zeros(SAMPLE_NUM, dtype = np.float32)
    y_vec = np.zeros(SAMPLE_NUM, dtype = np.float32)
    for i in tqdm(range(SAMPLE_NUM)):
        xy = s.sample()
        x_vec[i] = xy[0, 0]
        y_vec[i] = xy[0, 1]
    
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