#coding: utf-8
import os, shutil
from pathlib import Path
import numpy as np
from PIL import Image
import json
import jax

'''
[Labels]
road
terrain
sidewalk
persongroup
vegetation
sky
traffic sign
pole
building
traffic light
ego vehicle
rectification border
out of roi
road
car
cargroup
traffic sign
vegetation
pole
building
sky
person
sidewalk
fence
ego vehicle
out of roi
road
car
cargroup
traffic sign
building
sky
vegetation
person
pole
sidewalk
ego vehicle
rectification border
out of roi
ego vehicle
out of roi
'''

class CityScapes:

    def __init__(self, base_path, rng, img_h = 1024, img_w = 2048):
        
        # initialize
        self.__rng = rng
        self.__all = {}
        
        # json
        for file_path in Path(base_path).rglob('*.json'):
            file_name = file_path.name
            train_type = file_path.parents[1].name
            file_key = None
            for ext in ['_gtCoarse_polygons.json',
                        '_gtFine_polygons.json']:
                if 0 < file_name.find(ext):
                    file_key = file_name[:file_name.rfind(ext)]
            assert(file_key != None)
            if train_type in self.__all.keys():
                if not file_key in self.__all[train_type].keys():
                    self.__add_key('json', file_path, train_type, file_key)
                else:
                    if file_name.find('_gtFine_polygons.json'): # fineなら上書き
                        self.__add_key('json', file_path, train_type, file_key)
            else:
                self.__add_key("json", file_path, train_type, file_key)
        # left
        for file_path in Path(base_path).rglob('*_leftImg8bit.png'):
            file_name = file_path.name
            train_type = file_path.parents[1].name
            file_key = None
            for ext in ['_leftImg8bit.png']:
                if 0 < file_name.find(ext):
                    file_key = file_name[:file_name.rfind(ext)]
            assert(file_key != None)
            self.__add_key("left", file_path, train_type, file_key)
        
        self.__img_h = img_h
        self.__img_w = img_w
        
    # initializetool function
    def __add_key(self, file_type, file_path, train_type, file_key):
        if not train_type in self.__all.keys():
            self.__all[train_type] = {}
        if not file_key in self.__all[train_type].keys():
            self.__all[train_type][file_key] = {}
        self.__all[train_type][file_key][file_type] = file_path
    
    def __json_to_rects(self, json_path, label_list):
        label_dict = {}
        info = json.load(open(json_path, "r", encoding="utf-8_sig"))
        img_h = info["imgHeight"]
        img_w = info["imgWidth"]
        obj_list = info["objects"]
        for obj in obj_list:
            obj_label = obj["label"]
            if obj_label in label_list:
                polygons = np.array(obj["polygon"])
                x0 = np.clip(polygons[:,0].min() / (img_w - 1), 0, 1.0)
                x1 = np.clip(polygons[:,0].max() / (img_w - 1), 0, 1.0)
                y0 = np.clip(polygons[:,1].min() / (img_h - 1), 0, 1.0)
                y1 = np.clip(polygons[:,1].max() / (img_h - 1), 0, 1.0)
                yc = (y0 + y1) / 2
                xc = (x0 + x1) / 2
                h  = y1 - y0
                w  = x1 - x0
                rect = np.array([yc, xc, h, w]).reshape(1, 4)
                if not obj_label in label_dict.keys():
                    label_dict[obj_label] = rect
                else:
                    label_dict[obj_label] = np.append(label_dict[obj_label], rect, axis = 0)
        return label_dict
    
    def itrnum_in_epoch(self, train_type, batch_size):
        train_type_dict = self.__all[train_type]
        key_list = list(train_type_dict.keys())
        n_data = len(key_list)
        return n_data // batch_size

    def make_generator(self, train_type, label_txt_list, batch_size,
                        aug_flip = False,
                        aug_crop_y0 = 0.0,
                        aug_crop_y1 = 1.0,
                        aug_crop_x0 = 0.0,
                        aug_crop_x1 = 1.0,
                        ):
        train_type_dict = self.__all[train_type]
        key_list = list(train_type_dict.keys())
        n_data = len(key_list)
        
        while True:
            images = np.empty(0, dtype = np.float32)
            labels = []
            
            self.__rng, rng = jax.random.split(self.__rng)
            for i in jax.random.randint(rng, (batch_size,), 0, n_data):
                # selected index of image
                key = key_list[i]
                tgt = train_type_dict[key]
                assert("left" in tgt.keys())

                # fix augumentation setting
                flip = False
                if aug_flip:
                    self.__rng, rng = jax.random.split(self.__rng)
                    flip = jax.random.randint(rng, (1,), 0, 2, jnp.bool)
                self.__rng, rng = jax.random.split(self.__rng)
                crop_y0 = jax.random.uniform(rng) * (aug_crop_y0 - 0.0) + 0.0
                self.__rng, rng = jax.random.split(self.__rng)
                crop_y1 = jax.random.uniform(rng) * (1.0 - aug_crop_y1) + aug_crop_y1
                self.__rng, rng = jax.random.split(self.__rng)
                crop_x0 = jax.random.uniform(rng) * (aug_crop_x0 - 0.0) + 0.0
                self.__rng, rng = jax.random.split(self.__rng)
                crop_x1 = jax.random.uniform(rng) * (1.0 - aug_crop_x1) + aug_crop_x1

                if len(label_txt_list) > 0:
                    if "json" in tgt.keys():
                        label_info = self.__json_to_rects(tgt["json"], label_txt_list)
                    else:
                        label_info = {}
                        for label in label_txt_list:
                            label_info[label] = []
    
                left_path = tgt["left"]
                left_pil = Image.open(left_path)
                org_w, org_h = left_pil.shape
                # crop augument
                left_pil = left_pil.crop((int(crop_x0 * (org_w - 1)),
                                          int(crop_y0 * (org_h - 1)),
                                          int(crop_x1 * (org_w - 1)),
                                          int(crop_y1 * (org_h - 1))))
                left_pil = left_pil.resize((self.__img_w, self.__img_h))
                left_arr = np.asarray(left_pil)
                # flip augument
                assert(left_arr.ndim == 3)
                if flip:
                    left_arr = left_arr[:,::-1,:]
                left_arr = left_arr.reshape(   (1,
                                                left_arr.shape[0],
                                                left_arr.shape[1],
                                                left_arr.shape[2]))
                if images.size == 0:
                    images = left_arr
                else:
                    images = np.append(images, left_arr, axis = 0)
                labels.append(label_info)
            
            yield images.astype(np.float32), labels
            
def visualize(data_path, dst_dir_path, color):
    from PIL import ImageDraw, ImageFont
    rng_key = jax.random.PRNGKey(0)
    cityscapes = CityScapes(data_path, rng_key, 256, 512)
    epoch_num = 10
    batch_size = 2
    gen = cityscapes.make_generator("train",
                                    label_txt_list = list(color.keys()),
                                    batch_size = batch_size)
    for e in range(epoch_num):
        images, labels = next(gen)
        for b in range(batch_size):
            left_arr, label_dict = images[b], labels[b]
            pil = Image.fromarray(left_arr.astype(np.uint8))
            img_w, img_h = pil.size
            dr = ImageDraw.Draw(pil)
            for label_name, rect_arr in label_dict.items():
                for rect in rect_arr:
                    yc = rect[0]
                    xc = rect[1]
                    h  = rect[2]
                    w  = rect[3]

                    y0 = yc - h / 2
                    y1 = yc + h / 2
                    x0 = xc - w / 2
                    x1 = xc + w / 2
                    
                    x0 *= (img_w - 1)
                    x1 *= (img_w - 1)
                    xc *= (img_w - 1)
                    y0 *= (img_h - 1)
                    y1 *= (img_h - 1)
                    yc *= (img_h - 1)
                    dr.rectangle((x0, y0, x1, y1), width = 3, outline = color[label_name])
                    dr.text((xc, yc), label_name, fill = color[label_name])
            dst_path = os.path.join(dst_dir_path, "{}_{}.jpg".format(e, b))
            pil.save(dst_path)
            print(dst_path)

if __name__ == "__main__":
    data_path = r"/mnt/hdd/dataset/cityscapes"
    dst_dir_path = r"visualize"
    
    if os.path.exists(dst_dir_path):
        shutil.rmtree(dst_dir_path)
    os.makedirs(dst_dir_path)

    visualize(data_path, dst_dir_path, color = {"car":(255,0,0), "person":(0,255,0)})
    
