import os
import random

'''
用于生成VOC风格的目标检测数据集分布  txt格式
trainval.txt   
train.txt   训练集
val.txt     验证集
test.txt    测试集

input： train和val所占比例trainval_percent    train占trainval的比例train_percent   标注文件位置   输出的txt位置
output: 四个txt 
'''

trainval_percent = 0.95
train_percent = 0.95
xmlfilepath = 'G:/data_trackdisease/test_folder/xml_out'     #标注数据位置
txtsavepath = 'G:/data_trackdisease/test_folder/imgset'      #输出的txt位置




class Distribution_dataset():
    def __init__(self):
        pass


    def disribute(self,trainval_percent,train_percent,xmlfilepath,txtsavepath):
        self.trainval_percent = trainval_percent
        self.train_percent = train_percent
        self.xmlfilepath = xmlfilepath
        self.txtsavepath = txtsavepath

        if os.path.isdir(self.txtsavepath) == False:
            os.makedirs(self.txtsavepath)


        total_xml = os.listdir(self.xmlfilepath)


        self.num=len(total_xml)
        list=range(self.num)
        self.tv=int(self.num*self.trainval_percent)
        print("tv:"+str(self.tv))
        self.tr=int(self.tv*self.train_percent)
        print("tr:" + str(self.tr))
        trainval= random.sample(list,self.tv)
        train=random.sample(trainval,self.tr)

        ftrainval = open(self.txtsavepath+'/trainval.txt', 'w')
        ftest = open(self.txtsavepath+'/test.txt', 'w')
        ftrain = open(self.txtsavepath+'/train.txt', 'w')
        fval = open(self.txtsavepath+'/val.txt', 'w')

        for i  in list:
            name=total_xml[i][:-4]+'\n'
            if i in trainval:
                ftrainval.write(name)
                if i in train:
                    ftrain.write(name)
                else:
                    fval.write(name)
            else:
                ftest.write(name)

        ftrainval.close()
        ftrain.close()
        fval.close()
        ftest .close()

if __name__ == '__main__':
    dist_data=Distribution_dataset()
    dist_data.disribute(trainval_percent,train_percent,xmlfilepath,txtsavepath)
    print("mission complete")
    print("total num: {} , train num:{} , test num:{} , val num: {}".format(dist_data.num,dist_data.tr,dist_data.num-dist_data.tv,dist_data.tv-dist_data.tr))