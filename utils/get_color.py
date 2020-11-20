import cv2
import cclib
import numpy as np
from tqdm import tqdm

from collections import Counter
import json


def get_color(img_path, id_path):
    input = open(id_path, 'r', encoding="utf-8")
    item_color = []
    exist = []
    try:
        file = open('./data/item_color.txt', 'r')
        for i in file:
            aa = i.split('-')[0]
            exist.append(aa)
    except:
        pass


    for idx, line in enumerate(input):
        rs = line.replace('\n', '')
        filename = rs + '.jpg'
        img = cv2.imread(img_path + filename)
        if rs in exist:
            continue

        if img is None:
            str2 = filename + ' ' + ' \n'
            print('%s failed' % str2)
        else:
            img = cclib.resizeImg(img, 0.3)
            color = cclib.getClothesColor2(img)[0]
            # colorHSV = cclib.getClothesColor2(img)[1]
            log = rs+'-'+color
            print(idx, log)
            file = open('./data/item_color.txt', 'a+')
            file.write(log+'\n')
            file.close()
            str2 = rs + '\t' + color
            item_color.append(str2)


# get_color('D:/for11k/items/', './item_id.txt')