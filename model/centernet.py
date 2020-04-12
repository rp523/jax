#coding: utf-8

from .maker.model_maker import net_maker
import jax
import jax.numpy as jnp
import numpy as onp

def make_heatmap(labels, h, w, all_texts):
    batch_size = len(labels)
    n_class = len(all_texts)
    heatmap = jnp.zeros((batch_size, h, w, n_class))
    for b, label in enumerate(labels):
        for label_name, rects in label.items():
            label_idx = all_texts.index(label_name)
            for rect in rects:
                x0 = rect[0]
                x1 = rect[1]
                y0 = rect[2]
                y1 = rect[3]
                cx = (x0 + x1) / 2
                cy = (y0 + y1) / 2
                sx = x1 - x0
                sy = y1 - y0
                dx = (jnp.arange(w) - w * cx) / w / (sx / 2)
                dy = (jnp.arange(h) - h * cy) / h / (sy / 2)
                dx = jnp.tile(dx, (h, 1))
                dy = jnp.tile(dy.reshape(-1, 1), (1, w))
                d2 = (dx ** 2 + dy ** 2)
                rect_map = jnp.exp(- d2)
                heatmap  = jax.ops.index_max(heatmap, jax.ops.index[b,:,:,label_idx], rect_map)
    return heatmap

class debug:
    def __init__(self):
        pass
    def __call__(self, images, labels, all_texts, cnt):
        def make_debug_map(image, heatmap):
            assert(image.ndim == 3)
            assert(heatmap.ndim == 2)
            from PIL import Image
            norm_r = jnp.ones(heatmap.shape)
            norm_g = jnp.ones(heatmap.shape) - heatmap
            norm_b = jnp.ones(heatmap.shape) - heatmap
            norm_rgb = jnp.array([norm_r, norm_g, norm_b]).transpose((1, 2, 0))
            rgb = jnp.clip(norm_rgb * 255, 0, 255).astype(jnp.uint8)
            label_rate = 0.5
            all = (1.0 - label_rate) * image + label_rate * rgb
            all = onp.array(all)
            return Image.fromarray(all.astype(jnp.uint8))
        image_shape = images.shape
        h = image_shape[1]
        w = image_shape[2]
        heatmaps = make_heatmap(labels, h, w, all_texts)
        for b, image in enumerate(images):
            for c, cls in enumerate(all_texts):
                pil = make_debug_map(image, heatmaps[b,:,:,c])
                pil.save("img/{}_{}_{}.png".format(cnt,b, cls))
        
def main():
    pass

if __name__ == "__main__":
    main()
    print("Done.")
