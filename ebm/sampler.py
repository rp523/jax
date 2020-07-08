#coding: utf-8
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm
PROB_TYPE = "center_wave"

class Sampler:
    def __init__(self, rng, batch_size, half_band):
        self.__maker = self.__Maker(rng, batch_size, half_band)
        self.__half_band = half_band

        bin_num = max(batch_size * 2, 256)
        x = jnp.linspace(-half_band, half_band, bin_num)
        x = jnp.tile(x.reshape(1, -1), (bin_num, 1))
        y = jnp.linspace(-half_band, half_band, bin_num)
        y = jnp.tile(y.reshape(-1, 1), (1, bin_num))
        data = jnp.append(x.reshape(-1, 1), y.reshape(-1, 1), axis = 1)
        assert(data.shape == (bin_num * bin_num, 2))
        unnorm_map = self.__unnorm_prob(data)
        self.__unnorm_max_val = unnorm_map.max()
        self.__norm = unnorm_map.sum() * ((2 * half_band / bin_num) ** 2)
    def sample(self):
        return next(self.__maker)

    def __Maker(self, rng, batch_size, half_band):

        split_num = 8
        xy = jnp.zeros((batch_size, 2), dtype = jnp.float32)
        sampled_num = 0
        while True:
            rng_x, rng_p, rng = jax.random.split(rng, 3)
            x = (jax.random.uniform(rng_x, (split_num, 2)) * 2 - 1) * half_band
            p = jax.random.uniform(rng_p, (split_num,)) * self.__unnorm_max_val
            is_valid = (p < self.__unnorm_prob(x))
            assert(is_valid.shape == (split_num,))
            valid_num = is_valid.sum()
            if valid_num > 0:
                valid_indexes = jnp.where(is_valid)[0]
                assert(valid_indexes.size == valid_num)
                remain_num = batch_size - sampled_num
                add_num = min(remain_num, valid_num)
                xy = jax.ops.index_update(xy, jax.ops.index[sampled_num:sampled_num + add_num], x[is_valid][:add_num])
                sampled_num += add_num
                if sampled_num >= batch_size:
                    yield xy
                    sampled_num = 0
    
    # normalized probability density
    def prob(self, x):
        return self.__unnorm_prob(x) / self.__norm

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

def exect_plot():
    bin_num = 50
    half_band = 1.0
    x = jnp.linspace(-half_band, half_band, bin_num)
    x = jnp.tile(x.reshape(1, -1), (bin_num, 1))
    y = jnp.linspace(-half_band, half_band, bin_num)
    y = jnp.tile(y.reshape(-1, 1), (1, bin_num))
    data = jnp.append(x.reshape(-1, 1), y.reshape(-1, 1), axis = 1)
    assert(data.shape == (bin_num * bin_num, 2))
    rng = jax.random.PRNGKey(0)
    sampler = Sampler(rng, bin_num ** 2, half_band)
    z = sampler.prob(data)
    print(z.mean())
    assert(z.shape == (bin_num * bin_num,))
    z = z.reshape((bin_num, bin_num))
    X = jnp.linspace(-half_band, half_band, bin_num)
    Y = jnp.linspace(-half_band, half_band, bin_num)
    X, Y = jnp.meshgrid(X, Y)
    plt.pcolor(X, Y, z)
    plt.colorbar()
    plt.show()


def main():
    rng = jax.random.PRNGKey(0)
    SAMPLE_NUM = 100000
    BATCH_SIZE = 1000
    half_band = 1
    s = Sampler(rng, batch_size = BATCH_SIZE, half_band = half_band)
    x_vec = jnp.zeros(SAMPLE_NUM, dtype = jnp.float32)
    y_vec = jnp.zeros(SAMPLE_NUM, dtype = jnp.float32)
    for i in tqdm(range(SAMPLE_NUM // BATCH_SIZE)):
        xy = s.sample()
        x_vec = jax.ops.index_update(x_vec, jax.ops.index[i * BATCH_SIZE:(i + 1) * BATCH_SIZE], xy[:, 0])
        y_vec = jax.ops.index_update(y_vec, jax.ops.index[i * BATCH_SIZE:(i + 1) * BATCH_SIZE], xy[:, 1])

    fig = plt.figure()
    ax = fig.add_subplot(111)

    H = ax.hist2d(x_vec, y_vec, bins=40, cmap=cm.jet)
    ax.set_title('1st graph')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    fig.colorbar(H[3],ax=ax)
    plt.show()

if __name__ == "__main__":
    #exect_plot();exit()
    main()
    print("Done.")