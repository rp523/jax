#coding: utf-8
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm
PROB_TYPE = "center_wave"
cenwav_param = [[0.5, 0.1]]

class Sampler:
    def __init__(self, rng, batch_size, half_band):
        self.__rng = rng
        self.__batch_size = batch_size
        self.__half_band = half_band

    def sample(self):
        x = jnp.zeros((self.__batch_size, 2))
        if PROB_TYPE == "center_wave":
            for cr, s in cenwav_param:
                cr = cr * self.__half_band
                s = s * self.__half_band
                rng_r, rng_t, self.__rng = jax.random.split(self.__rng, 3)
                rawr = jax.random.normal( rng_r, (self.__batch_size,))
                rawt = jax.random.uniform(rng_t, (self.__batch_size,))
                r = jnp.abs(cr + rawr * s)
                t = rawt * (2 * jnp.pi)
                x0 = r * jnp.cos(t)
                x1 = r * jnp.sin(t)
                x += jnp.append(x0.reshape((-1, 1)), x1.reshape((-1, 1)), axis = 1)
        return x
    

    def prob(self, x):
        if PROB_TYPE == "center_wave":
            for cr, s in cenwav_param:
                cr = cr * self.__half_band
                s = s * self.__half_band
                power = - ((((x ** 2).sum(axis = -1) ** 0.5) - cr) ** 2) / (2 * (s ** 2))
                p = jnp.exp(power)
        return p

    # unnormalized probability density
    def __unnorm_prob(self, x):
        assert(x.shape[-1] == 2)

        if PROB_TYPE == "center_wave":
            delta_r = 0.5 * self.__half_band
            sigma = 0.2 * self.__half_band

            cx = 0.0
            cy = 0.0

            rx = x.T[0] - cx
            ry = x.T[1] - cy
            r = (rx ** 2 + ry ** 2) ** 0.5

            ret = jnp.exp(- ((r - delta_r) ** 2) / (2 * sigma ** 2))
        elif PROB_TYPE == "block":
            ret = 0.0
            split_num = 5
            sigma = 1.0 / split_num / 2.0
            for i in range(split_num):
                for j in range(split_num):
                    if (i + j) % 2 == 0:
                        cx = (i + 0.5) / split_num - 0.5
                        cy = (j + 0.5) / split_num - 0.5
                        rx = x.T[0] - cx
                        ry = x.T[1] - cy
                        ret += jnp.exp(- (rx ** 2 + ry ** 2) / (2 * sigma ** 2))
            ret = ret / 0.7025556   # normalize so that integral is unity. max is 1.5295591
        elif PROB_TYPE == "triangle":
            ret = jnp.ones(x.shape[0])
            angle_num = 3
            sigma = 0.02
            delta = 0.25
            for i in range(angle_num):
                theta = 2 * jnp.pi * (i / angle_num)
                cos_val = jnp.cos(theta)
                sin_val = jnp.sin(theta)
                rot_mat = jnp.array(   [[cos_val, -sin_val],
                                        [sin_val,  cos_val]])
                lined_x = -jnp.dot(rot_mat, x.T)[1]
                weight = jax.nn.sigmoid(-(lined_x - delta) / sigma)
                ret *= (weight)
        return ret

def main():
    rng = jax.random.PRNGKey(0)
    SAMPLE_NUM = 1000000
    BATCH_SIZE = 1000
    half_band = 5
    bin_num = 100
    s = Sampler(rng, batch_size = BATCH_SIZE, half_band = half_band)
    x_vec = jnp.zeros(SAMPLE_NUM, dtype = jnp.float32)
    y_vec = jnp.zeros(SAMPLE_NUM, dtype = jnp.float32)
    for i in tqdm(range(SAMPLE_NUM // BATCH_SIZE)):
        xy = s.sample()
        x_vec = jax.ops.index_update(x_vec, jax.ops.index[i * BATCH_SIZE:(i + 1) * BATCH_SIZE], xy[:, 0])
        y_vec = jax.ops.index_update(y_vec, jax.ops.index[i * BATCH_SIZE:(i + 1) * BATCH_SIZE], xy[:, 1])

    fig = plt.figure(figsize = (10, 5))

    ax = fig.add_subplot(121)
    H = ax.hist2d(x_vec, y_vec, bins = bin_num, cmap=cm.jet)
    ax.set_title('1st graph')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    fig.colorbar(H[3],ax=ax)

    ax = fig.add_subplot(122)
    x = jnp.linspace(-half_band, half_band, bin_num)
    x = jnp.tile(x.reshape(1, -1), (bin_num, 1))
    y = jnp.linspace(-half_band, half_band, bin_num)
    y = jnp.tile(y.reshape(-1, 1), (1, bin_num))
    data = jnp.append(x.reshape(-1, 1), y.reshape(-1, 1), axis = 1)
    X = jnp.linspace(-half_band, half_band, bin_num)
    Y = jnp.linspace(-half_band, half_band, bin_num)
    X, Y = jnp.meshgrid(X, Y)
    plt.pcolor(X, Y, s.prob(data).reshape((bin_num, bin_num)))
    plt.colorbar()

    plt.show()

if __name__ == "__main__":
    #exect_plot();exit()
    main()
    print("Done.")