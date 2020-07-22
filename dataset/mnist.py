#coding: utf-8
import os, gzip
import numpy as np
import urllib.request
from PIL import Image
import jax
import jax.numpy as jnp

class Mnist:
    def __init__(self, rng, batch_size, data_type, one_hot, dequantize, flatten, remove_classes = None, remove_col_too = False):
        self.__batch_size = batch_size
        self.__rng = rng
        self.__data_type = data_type

        url_base = "http://yann.lecun.com/exdb/mnist/"
        self.__key_file = {
            "train_img":"train-images-idx3-ubyte.gz",
            "train_label":"train-labels-idx1-ubyte.gz",
            "test_img":"t10k-images-idx3-ubyte.gz",
            "test_label":"t10k-labels-idx1-ubyte.gz"
        }
        dir_path = "mnist"
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        for file_name in self.__key_file.values():
            file_path = os.path.join(dir_path, file_name)
            if not os.path.exists(file_path):
                urllib.request.urlretrieve(url_base + file_name, file_path)

        self.__all_data = {}
        for key, file_name in self.__key_file.items():
            file_path = os.path.join(dir_path, file_name)
            with gzip.open(file_path, 'rb') as f:
                data = np.frombuffer(f.read(), np.uint8)
                if key.find("img") >= 0:
                    data = data[16:].reshape((-1, 28, 28))
                    if dequantize:
                        self.__rng, _rng = jax.random.split(self.__rng)
                        data = Mnist.dequantize(_rng, data)
                    if flatten:
                        data = data.reshape((data.shape[0], -1))
                elif key.find("label") >= 0:
                    data = data[8:].flatten()
                else:
                    assert(0)
                self.__all_data[key] = data

        if remove_classes is not None:
            for img_key, lbl_key in    [["train_img", "train_label"],
                                        ["test_img", "test_label"]]:
                for remove_val in remove_classes:
                    is_remain = (self.__all_data[lbl_key] != remove_val)
                    self.__all_data[lbl_key] = self.__all_data[lbl_key][is_remain]
                    self.__all_data[img_key] = self.__all_data[img_key][is_remain]

        if one_hot:
            for lbl_key in ["train_label", "test_label"]:
                self.__all_data[lbl_key] = jnp.eye(10)[self.__all_data[lbl_key]]
                if remove_classes is not None:
                    if remove_col_too:
                        for remove_val in np.sort(np.asarray(remove_classes))[::-1]:
                            self.__all_data[lbl_key] = np.delete(self.__all_data[lbl_key], obj = remove_val, axis = 1)
        
        self.__class_num = 10
        if remove_classes is not None:
            self.__class_num -= len(remove_classes)
        
    def test_visualize(self):
        count = {}
        labels = np.arange(self.__class_num)
        for label in labels:
            count[int(label)] = 0
        for img_key, label_key in [ ["train_img", "train_label"],
                                    ["test_img",  "test_label"]]:
            dst_dir_path = img_key[:img_key.find("_")]
            if not os.path.exists(dst_dir_path):
                os.makedirs(dst_dir_path)
            imgs = self.__all_data[img_key]
            labels = self.__all_data[label_key]
            for img, label in zip(imgs, labels):
                pil = Image.fromarray(img.astype(np.uint8))
                dst_name = "{0:05d}".format(label) + "_{}.png".format(count[int(label)])
                dst_path = os.path.join(dst_dir_path, dst_name)
                pil.save(dst_path)
                print(dst_path)
                count[int(label)] += 1

    def sample(self, get_all = False):
        if self.__data_type == "train":
            imgs = self.__all_data["train_img"]
            lbls = self.__all_data["train_label"]
        elif self.__data_type == "test":
            imgs = self.__all_data["test_img"]
            lbls = self.__all_data["test_label"]
        data_num = imgs.shape[0]

        if get_all is False:
            self.__rng, _rng = jax.random.split(self.__rng)
            sel_idxs = jax.random.randint(_rng, (self.__batch_size,), 0, data_num)
            sel_idxs = np.array(sel_idxs)

            sel_imgs = imgs[sel_idxs]
            sel_lbls = lbls[sel_idxs]
        else:
            sel_imgs = imgs
            sel_lbls = lbls
        
        return sel_imgs, sel_lbls 
    
    @staticmethod
    def dequantize(rng, x):
        x = x.astype(np.float32) + jax.random.uniform(rng, x.shape)
        x = jnp.clip(x / 256, 0.0, 1.0)
        eps = 1E-10
        return jnp.log(x + eps) - jnp.log(1 - x + eps)

    @staticmethod
    def quantize(x):
        x = jax.nn.sigmoid(x)
        x = jnp.clip(x * 255, 0, 255).astype(jnp.uint8)
        return x

def main():
    rng = jax.random.PRNGKey(0)
    m = Mnist(  rng,
                batch_size = 1,
                data_type = "train",
                one_hot = False,
                dequantize = True,
                flatten = False,
    )
    imgs, lbls = m.sample()
    print(imgs.min(), imgs.max())
    imgs = Mnist.quantize(imgs)
    imgs = np.asarray(jax.device_put(imgs))
    print(imgs.min(), imgs.max())
    img = imgs[0]
    Image.fromarray(img).show()
    print(lbls[0])

if __name__ == "__main__":
    main()
    print("Done.")