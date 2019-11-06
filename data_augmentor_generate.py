# -*- coding=utf-8 -*-

'''
#这个Augmentor包使用是有模板的，输入为一个文件夹的路径
文件夹中有以图像类别为名的子文件夹，其中包含了数据图片
包中默认是在子文件夹中添加一个output文件夹存放处理后的数据，改为由用户定义存放位置out_data_path
具体每个操作对图像的影响查询 Augmentor包的文档
程序中的概率可以自己设置
'''
import Augmentor

import shutil,os

width_img=280    #预设大小是为了处理图片更快，例如4000*4000的原图 如果只想得到224*224的图，不妨先设置到280*280做处理，最后再resize到224
out_width_img=224   #处理后输出图像大小
data_path = 'G:/data_xiaoyanta/data/voc'  # 读取文件夹位置    有n个子文件夹，包含了待处理图片，n为图像类别数
out_data_path = 'G:/data_xiaoyanta/data/voc300'  # 输出 文件夹位置
num_sample=2000  #每一类图片希望扩增到的数量（不包括原图）


if  __name__ == '__main__':

    classes_list = os.listdir(data_path)

    if os.path.isdir(out_data_path) == False:
        os.makedirs(out_data_path)
    for class_name in classes_list:
        sub_classes_path_lite = os.path.join(data_path, str(class_name))
        temp_classes_list = os.listdir(sub_classes_path_lite)
        if len(temp_classes_list)==0:
            continue

        p = Augmentor.Pipeline(sub_classes_path_lite)

        p.resize(probability=1, height=width_img, width=width_img)     #先规定到预设大小
        p.random_erasing(probability=0.1, rectangle_area=0.15)    # 擦除一块的 概率与大小
        p.skew(probability=0.18, magnitude=0.1)      #视角变换概率为 0.2  变换程度为0.1
        p.skew(probability=0.18, magnitude=0.2)
        p.rotate(probability=0.15, max_left_rotation=5, max_right_rotation=5)    #旋转倾斜  概率为0.2
        #0.15 的概率 会被裁剪为边长是以前的 0.95 0.9 0.85  0.8 0.75
        p.crop_by_size(probability=0.2, width=width_img * 0.95, height=width_img * 0.95, centre=False)
        p.crop_by_size(probability=0.2, width=width_img * 0.9, height=width_img * 0.9, centre=False)
        p.crop_by_size(probability=0.2, width=width_img * 0.85, height=width_img * 0.85, centre=False)
        p.crop_by_size(probability=0.2, width=width_img * 0.8, height=width_img * 0.8, centre=False)


        p.resize(probability=1, height=out_width_img, width=out_width_img)     #规定为统一输出大小
        p.sample(num_sample)   #一个文件夹一共获得多少样本



        thispath=os.path.join(sub_classes_path_lite,'output')
        outpath=os.path.join(out_data_path,class_name)
        shutil.move(thispath,outpath)
        print("mission complete")
        print("process "+str(len(classes_list))+" classes image")


