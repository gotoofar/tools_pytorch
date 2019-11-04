# -*- coding=utf-8 -*-

'''
该程序用于已标注图像的增广
输入的路径是存放图像的文件夹 img_folder 和 存放标注信息的文件夹 xml_folder (仅支持xml)
AugmentDetection()类包含了几个方法，每个方法都定义了一种处理手段

_crop_img_bboxes 裁剪
_filp_pic_bboxes 翻转/镜像
_cutout 随机盖住一个或多个块
_changeLight  调整亮度  因为调用的包出了错误 暂时没有弄上
_addNoise    加噪声  同样原因没有弄上


'''
import time
import random
import cv2
import os
import math
import numpy as np
from tqdm import tqdm
#from skimage.util import random_noise
#from skimage import exposure


need_aug_num = 5   #需要增广的图像数量
#输入数据位置
source_pic_root_path ='G:/data_trackdisease/test_folder/img_test'
source_xml_root_path = 'G:/data_trackdisease/test_folder/xml_test'

#输出数据位置
out_xml = 'G:/data_trackdisease/test_folder/xml_out'
out_img = 'G:/data_trackdisease/test_folder/img_out'


class AugmentDetection():
    #设定增强系数
    def __init__(self,
                 change_light_rate=0.5,    #改变亮度概率
                 add_noise_rate=0.5,       #加噪声概率
                 flip_rate=0.67,           #翻转概率
                 cutout_rate=0,            #遮盖概率
                 cut_out_length=20,        #遮盖大小
                 cut_out_holes=1,          #遮盖数
                 cut_out_threshold=0.5,    #遮盖阈值
                 crop_size=0.9,            #裁切比例  裁切下来的长宽为  0.9~1 原图比
                 crop_rate=0.8,            #裁切概率
                 resize_img=1,             #是否缩放
                 resize_param=1.0/2.0):    #缩放比例



        self.change_light_rate = change_light_rate
        self.add_noise_rate = add_noise_rate
        self.flip_rate = flip_rate
        self.cutout_rate = cutout_rate
        self.cut_out_length = cut_out_length
        self.cut_out_holes = cut_out_holes
        self.cut_out_threshold = cut_out_threshold
        self.crop_size=crop_size
        self.crop_rate=crop_rate
        self.resize_img=resize_img
        self.resize_param=resize_param



    def _resize_img_bboxes(self,img,bboxes):
        w = img.shape[1]
        h = img.shape[0]
        resize_width=int(w*self.resize_param)
        resize_height=int(h*self.resize_param)

        img_resize=cv2.resize(img,(resize_width,resize_height))

        resize_bboxes = list()
        for bbox in bboxes:
            resize_bboxes.append([bbox[0]*self.resize_param, bbox[1]*self.resize_param, bbox[2]*self.resize_param, bbox[3]*self.resize_param, bbox[4]])

        return img_resize, resize_bboxes

    # cutout
    def _cutout(self, img, bboxes, length, n_holes=1, threshold=0.5):
        '''

        Randomly mask out one or more patches from an image.
        随机抠掉一块用全0代替
        如果抠掉的部分和目标重叠太多则重新找位置
        Args:
            img : a 3D numpy array,(h,w,c)
            bboxes : 框的坐标
            n_holes (int): Number of patches to cut out of each image.
            length (int): The length (in pixels) of each square patch.
        '''

        def cal_iou(boxA, boxB):
            '''
            boxA, boxB为两个框，返回iou
            boxB为bouding box
            '''

            # determine the (x, y)-coordinates of the intersection rectangle
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[2], boxB[2])
            yB = min(boxA[3], boxB[3])

            if xB <= xA or yB <= yA:
                return 0.0

            # compute the area of intersection rectangle
            interArea = (xB - xA + 1) * (yB - yA + 1)

            boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
            boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

            # compute the intersection over union by taking the intersection
            # area and dividing it by the sum of prediction + ground-truth
            # areas - the interesection area
            # iou = interArea / float(boxAArea + boxBArea - interArea)
            iou = interArea / float(boxBArea)

            return iou

        # 得到h和w
        if img.ndim == 3:
            h, w, c = img.shape
        else:
            _, h, w, c = img.shape

        mask = np.ones((h, w, c), np.float32)

        for n in range(n_holes):

            overlap = True  # 看切割的区域是否与box重叠太多 重叠太多就重新找

            while overlap:
                y = np.random.randint(h)
                x = np.random.randint(w)

                length_y=int(random.uniform(length*0.5,length))
                length_x=int(random.uniform(length*0.5,length))
                y1 = np.clip(y - length_y // 2, 0,
                             h)  # numpy.clip(a, a_min, a_max, out=None), clip这个函数将将数组中的元素限制在a_min, a_max之间，大于a_max的就使得它等于 a_max，小于a_min,的就使得它等于a_min
                y2 = np.clip(y + length_y // 2, 0, h)
                x1 = np.clip(x - length_x // 2, 0, w)
                x2 = np.clip(x + length_x // 2, 0, w)

                overlap = False
                for box in bboxes:
                    if cal_iou([x1, y1, x2, y2], box) > threshold:
                        overlap = True
                        break
            #置0
            mask[y1: y2, x1: x2, :] = 0.


        # mask = np.expand_dims(mask, axis=0)
        img = img * mask

        return img


    def _crop_img_bboxes(self, img, bboxes):
        '''
        裁剪后的图片要包含所有的框
        输入:
            img:图像array
            bboxes:该图像包含的所有boundingboxs,一个list,每个元素为[x_min, y_min, x_max, y_max],要确保是数值
        输出:
            crop_img:裁剪后的图像array
            crop_bboxes:裁剪后的bounding box的坐标list
        '''
        # ---------------------- 裁剪图像 ----------------------
        w = img.shape[1]
        h = img.shape[0]
        flag_empty=0   #是否裁出了一个没有目标的patch



        w_crop = int(random.uniform(min(w, h) * self.crop_size, min(w, h)))
        h_crop = int(random.uniform(min(w, h) * self.crop_size, min(w, h)))
        w_start = int(random.uniform(0, w - w_crop))
        h_start = int(random.uniform(0, h - h_crop))

        crop_img = img[h_start:h_start+h_crop, w_start:w_start+w_crop]

        x_min=w_start
        y_min=h_start
        x_max=w_start+w_crop
        y_max=h_start+h_crop


        # ---------------------- 裁剪boundingbox ----------------------
        # 裁剪后的boundingbox坐标计算
        crop_bboxes = list()
        for bbox in bboxes:
            if (bbox[3]<=y_min or bbox[1]>=y_max or bbox[0]>x_max or bbox[2]<x_min or (bbox[2]-x_min)<0.35*(bbox[2]-bbox[0]) or (x_max-bbox[0])<0.35*(bbox[2]-bbox[0]) or (bbox[3]-y_min)<0.35*(bbox[3]-bbox[1]) or (y_max-bbox[1])<0.35*(bbox[3]-bbox[1])   ):
                #print('error')
                continue
            elif(bbox[0]<x_min or bbox[1]<y_min or bbox[2]>x_max or bbox[3]>y_max):
                if(bbox[0]<x_min): bbox[0]=x_min
                if(bbox[1]<y_min): bbox[1]=y_min
                if(bbox[2]>x_max): bbox[2]=x_max
                if(bbox[3]>y_max): bbox[3]=y_max
                #print('change')

                crop_bboxes.append([bbox[0] - x_min, bbox[1] - y_min, bbox[2] - x_min, bbox[3] - y_min,bbox[4]])

            else:
                #print('ok')
                crop_bboxes.append([bbox[0] - x_min, bbox[1] - y_min, bbox[2] - x_min, bbox[3] - y_min, bbox[4]])

        if len(crop_bboxes)==0:  #如果裁切出来 里面一个目标都没有 则这个作废
            flag_empty=1
        return crop_img, crop_bboxes,flag_empty

    # 镜像
    def _filp_pic_bboxes(self, img, bboxes):
        '''

            输入:
                img:图像array
                bboxes:该图像包含的所有boundingboxs,一个list,每个元素为[x_min, y_min, x_max, y_max],要确保是数值
            输出:
                flip_img:平移后的图像array
                flip_bboxes:平移后的bounding box的坐标list
        '''
        # ---------------------- 翻转图像 ----------------------
        import copy
        flip_img = copy.deepcopy(img)
        if random.random() < 0.5:  # 水平和垂直翻转的概率都是0.5，二者可以叠加
            horizon = True
        else:
            horizon = False
        h, w, _ = img.shape
        if horizon:  # 水平翻转

            if random.random() <0.5:
                shuiping=True

            else:
                shuiping=False
            if shuiping:

                flip_img = cv2.flip(flip_img, 1)  #水平

            else:

                flip_img = cv2.flip(flip_img, 0)  # 上下
        else:
            flip_img=flip_img




        # ---------------------- 调整boundingbox ----------------------
        flip_bboxes = list()
        for box in bboxes:
            x_min = box[0]
            y_min = box[1]
            x_max = box[2]
            y_max = box[3]
            if horizon:
                if shuiping:

                    flip_bboxes.append([w - x_max, y_min, w - x_min, y_max,box[4]])
                else:
                    flip_bboxes.append([x_min, h - y_max, x_max, h - y_min,box[4]])
            else:

                flip_bboxes.append([x_min, y_min, x_max, y_max, box[4]])

        return flip_img, flip_bboxes



    # 加噪声
    # def _addNoise(self, img):
    #     '''
    #     输入:
    #         img:图像array
    #     输出:
    #         加噪声后的图像array,由于输出的像素是在[0,1]之间,所以得乘以255
    #     '''
    #     # random.seed(int(time.time()))
    #     # return random_noise(img, mode='gaussian', seed=int(time.time()), clip=True)*255
    #     return random_noise(img, mode='gaussian', clip=True) * 255

    # 调整亮度
    # def _changeLight(self, img):
    #     # random.seed(int(time.time()))
    #     flag = random.uniform(0.8, 1.2)  # flag>1为调暗,小于1为调亮
    #     return exposure.adjust_gamma(img, flag)


    def dataAugment(self, img, bboxes):

        change_num = 0  # 改变的次数



        while change_num < 1:  # 默认至少有一种数据增强生效

            # if random.random() < self.change_light_rate:  # 改变亮度
            #
            #     print('亮度')
            #     change_num += 1
            #     img = self._changeLight(img)

            #if random.random() < self.add_noise_rate:  # 加噪声
             #   print('加噪声')
             #   change_num += 1
               # img = self._addNoise(img)
            flag_empty=0
            if random.random()<self.crop_rate:
                img, bboxes, flag_empty = self._crop_img_bboxes(img, bboxes)
                change_num += 1


            if random.random() < self.cutout_rate:  # cutout

                change_num += 1
                img = self._cutout(img, bboxes, length=self.cut_out_length, n_holes=self.cut_out_holes,
                                   threshold=self.cut_out_threshold)

            if random.random() < self.flip_rate:  # 翻转

                change_num += 1
                img, bboxes = self._filp_pic_bboxes(img, bboxes)
        if self.resize_img:

            img, bboxes = self._resize_img_bboxes(img, bboxes)

        return img, bboxes,flag_empty



if __name__ == '__main__':



    import shutil
    from xml_helper import *



    dataAug = AugmentDetection()


    if os.path.isdir(out_img) == False:
        os.makedirs(out_img)
    if os.path.isdir(out_xml) == False:
        os.makedirs(out_xml)

    for parent, _, files in os.walk(source_pic_root_path):
        for file in tqdm(files):
            cnt = 0
            while cnt < need_aug_num:
                pic_path = os.path.join(parent, file)
                xml_path = os.path.join(source_xml_root_path, file[:-4] + '.xml')
                coords = parse_xml(xml_path)  # 解析得到box信息，格式为[[x_min,y_min,x_max,y_max,name]]

                img = cv2.imread(pic_path)
                auged_img, auged_bboxes,flag_empty = dataAug.dataAugment(img, coords)
                if flag_empty==1:

                    continue
                cnt += 1


                generate_xml(file[:-4]+'_'+str(cnt)+'.jpg', auged_bboxes, auged_img.shape, out_xml)
                cv2.imwrite(out_img+'/'+file[:-4]+'_'+str(cnt)+'.jpg',auged_img)

    print("mission complete,process "+str(len(files))+" pictures")
