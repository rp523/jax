#coding: utf-8
import os
from pathlib import Path
import numpy as np
from PIL import Image
import json
from jax.interpreters.batching import batch

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

    def __init__(self, base_path, img_h = 1024, img_w = 2048):
        
        # initialize
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
        rate_y = self.__img_h / img_h
        rate_x = self.__img_w / img_w
        obj_list = info["objects"]
        for obj in obj_list:
            obj_label = obj["label"]
            if obj_label in label_list:
                polygons = np.array(obj["polygon"])
                x0 = np.clip(polygons[:,0].min() * rate_x, 0, self.__img_w - 1)
                x1 = np.clip(polygons[:,0].max() * rate_x, 0, self.__img_w - 1)
                y0 = np.clip(polygons[:,1].min() * rate_y, 0, self.__img_h - 1)
                y1 = np.clip(polygons[:,1].max() * rate_y, 0, self.__img_h - 1)
                rect = np.array([x0, x1, y0, y1]).reshape(1, 4)
                if not obj_label in label_dict.keys():
                    label_dict[obj_label] = rect
                else:
                    label_dict[obj_label] = np.append(label_dict[obj_label], rect, axis = 0)
        return label_dict
    
    def make_generator(self, train_type, label_txt_list, batch_size, seed):
        np.random.seed(seed)
        train_type_dict = self.__all[train_type]
        key_list = list(train_type_dict.keys())
        n_data = len(key_list)
        
        images = np.empty(0)
        labels = []
        while True:
            for b in range(batch_size):
                i = np.random.randint(n_data)
                key = key_list[i]
                tgt = train_type_dict[key]
                assert("left" in tgt.keys())
                if len(label_txt_list) > 0:
                    if "json" in tgt.keys():
                        label_info = self.__json_to_rects(tgt["json"], label_txt_list)
                    else:
                        label_info = {}
                        for label in label_txt_list:
                            label_info[label] = []
    
                left_path = tgt["left"]
                left_pil = Image.open(left_path)
                left_pil = left_pil.resize((self.__img_w, self.__img_h))
                left_arr = np.asarray(left_pil)
                left_arr = left_arr.reshape((1,
                                             left_arr.shape[0],
                                             left_arr.shape[1],
                                             left_arr.shape[2]))
                if images.size == 0:
                    images = left_arr
                else:
                    images = np.append(images, left_arr, axis = 0)
                labels.append(label_info)
                
            yield images, labels
            
def visualize(data_path, dst_dir_path):
    from PIL import ImageDraw, ImageFont
    cityscapes = CityScapes(data_path, 256, 512)
    batch_size = 30
    gen = cityscapes.make_generator("train",
                                    label_txt_list = ["car", "person"],
                                    batch_size = batch_size,
                                    seed = 0)
    images, labels = next(gen)
    for b in range(batch_size):
        left_arr, label_dict = images[b], labels[b]
        pil = Image.fromarray(left_arr)
        dr = ImageDraw.Draw(pil)
        for label_name, rect_arr in label_dict.items():
            for rect in rect_arr:
                x0 = rect[0]
                x1 = rect[1]
                y0 = rect[2]
                y1 = rect[3]
                xc = (x0 + x1) // 2
                yc = (y0 + y1) // 2
                dr.rectangle((x0, y0, x1, y1), width = 3, outline = (255, 0, 0))
                dr.text((xc, yc), label_name, fill = (255, 0, 0))
        dst_path = os.path.join(dst_dir_path, "{}.jpg".format(b))
        pil.save(dst_path)
        print(dst_path)

if __name__ == "__main__":
    data_path = r"/mnt/hdd/dataset/cityscapes"
    dst_dir_path = r"."
    visualize(data_path, dst_dir_path)
    
