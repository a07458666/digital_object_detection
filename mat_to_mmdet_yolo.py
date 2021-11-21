import numpy as np
import os
import h5py
import cv2
from matplotlib import pyplot as plt
from tqdm import tqdm

FILE_NAME = "digitStruct.mat"
# The_Street_View_House_Numbers_Dataset
PATH_MAT = "../../DATA/andysu/SVHN/train/" + FILE_NAME


def get_name(index, hdf5_data):
    name_ref = hdf5_data["/digitStruct/name"][index].item()
    return "".join([chr(v[0]) for v in hdf5_data[name_ref]])


def get_bbox(index, data):
    attrs = {}
    item_ref = data["/digitStruct/bbox"][index].item()
    for key in ["label", "left", "top", "width", "height"]:
        attr = data[item_ref][key]
        values = (
            [data[attr[i].item()][0][0].astype(int) for i in range(len(attr))]
            if len(attr) > 1
            else [attr[0][0]]
        )
        attrs[key] = values
    return attrs


with h5py.File(PATH_MAT) as hdf5_data:
    for i in tqdm(range(33402)):
        img_name = get_name(i, hdf5_data)
        if not os.path.isfile(DATA_PATH_TRAIN + img_name):
            continue
        im = cv2.imread(DATA_PATH_TRAIN + img_name)
        h, w, c = im.shape
        fp = open(DATA_PATH_TRAIN + img_name.replace(".png", ".txt"), "w")
        arr = get_bbox(i, hdf5_data)
        #         print(arr)
        arr_l = len(arr["label"])
        annotations = []
        for idx in range(arr_l):
            label = arr["label"][idx]
            if label == 10:
                label = 0
            _l = arr["left"][idx]
            _t = arr["top"][idx]
            _w = arr["width"][idx]
            if (_l + _w) > w:
                _w = w - _l - 1
            _h = arr["height"][idx]
            if (_t + _h) > h:
                _h = h - _t - 1
            x_center = (_l + _w / 2) / w
            y_center = (_t + _h / 2) / h
            bbox_width = _w / w
            bbox_height = _h / h
            start_point = (
                int(w * (x_center - (bbox_width / 2))),
                int(h * (y_center - (bbox_height / 2))),
            )
            end_point = (
                int(w * (x_center + (bbox_width / 2))),
                int(h * (y_center + (bbox_height / 2))),
            )
            im = cv2.rectangle(im, start_point, end_point, color, thickness)
            s = (
                str(label)
                + " "
                + str(x_center)
                + " "
                + str(y_center)
                + " "
                + str(bbox_width)
                + " "
                + str(bbox_height)
            )
            if idx != (arr_l - 1):
                s += "\n"
            fp.write(s)
        fp.close()
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         plt.imshow(im)
#         plt.title(img_name)
#         plt.show()
