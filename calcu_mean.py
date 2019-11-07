# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np

'''
用于计算图像的各通道均值，作为图像预处理的参数
input: 包含图像的文件夹路径
output： 三通道均值
'''
path = 'G:/git_folder/ssd_data_and_weight/data and weight/VOC_stone/JPEGImages'


def compute(path):
    file_names = os.listdir(path)
    per_image_Rmean = []
    per_image_Gmean = []
    per_image_Bmean = []
    for file_name in file_names:
        img = cv2.imread(os.path.join(path, file_name), 1)
        per_image_Bmean.append(np.mean(img[:, :, 0]))
        per_image_Gmean.append(np.mean(img[:, :, 1]))
        per_image_Rmean.append(np.mean(img[:, :, 2]))
    R_mean = np.mean(per_image_Rmean)
    G_mean = np.mean(per_image_Gmean)
    B_mean = np.mean(per_image_Bmean)
    return R_mean, G_mean, B_mean


if __name__ == '__main__':
    R, G, B = compute(path)
    print(R, G, B)

