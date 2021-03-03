import numpy as np
import torch
import torchvision.transforms as T
from os import listdir
from os.path import join, isfile
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import random
import pickle
import pandas as pd

class RealGatesDS(torch.utils.data.Dataset):

    def __init__(self, dataset_path, pkl_path):

    def __len__(self):
        pass

    def get_height_and_width(self, index):
        pass

    def __getitem__(self, item):
        pass

if __name__ == '__main__':

    # ds = RealGatesDS(
    #     "/home/andreaalf/Documents/thesis/datasets/gate_full_sample",
    #     image_set='train',
    #     mask_rcnn=False
    # )

    ds = RealGatesDS("/home/andreaalf/Documents/thesis/datasets/gate_full_sample", 'train', backup_list_path='', mask_rcnn=True)

    for i, (img, target) in enumerate(ds):
        plt.imshow(img.cpu().permute(1, 2, 0))
        h, w = img.shape[-2:]
        bnd_box = target['boxes']
        for box in bnd_box:
            plt.scatter([box[2]*w, box[4]*w, box[6]*w], [box[3]*h, box[5]*h, box[7]*h])
            plt.scatter([box[0]*w], [box[1]*h])
        plt.show()

        # for mask in target['masks']:
        #     plt.imshow(mask)
        #     plt.show()
        # h, w = img.shape[-2:]
        #
        # boxes = target['boxes']
        # for gate in boxes:
        #     x = torch.tensor([
        #         gate[0],
        #         gate[2],
        #         gate[4],
        #         gate[6]
        #     ])
        #     y = torch.tensor([
        #         gate[1],
        #         gate[3],
        #         gate[5],
        #         gate[7]
        #     ])
        #     plt.scatter(x*w, y*h)
        #
        # plt.show()
