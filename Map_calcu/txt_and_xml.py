# -*- coding=utf-8 -*-
import os
import sys
import xml.etree.ElementTree as ET
import glob
from xml.etree.ElementTree import Element
from xml.etree import ElementTree
import xml.dom.minidom as xmldom
import numpy as np
import cv2
from functools import reduce


'''
该程序用于txt和xml的转换，配合这个map_calcu  包含xml2txt 和  txt2xml
txt格式: 每个框一行  【类别 四个坐标】        groundtruth没有置信度  predict里面有，在输出的时候把置信度放在第二个【类别 置信度 四个坐标】是给map calcu用的
xml格式为labelimg输出的标注文件

input:  要处理的文件夹的路径
output:  输出的文件夹路径


'''

class txt2xml(object):
    #感觉txt转xml应该用不上，如果得到输出的话可以直接分别输出xml和txt形式
    def __init__(self):
        pass
        self.name='txt to xml'

    def transform(self,txt_path,xml_path):
        file = []
        num_box_all=0
        txt_list = glob.glob(txt_path + '*.txt')
        txt_list.sort()

        for txt in txt_list:
            num_txt=len(txt_list)
            txt_id = txt.split('.txt', 1)[0]
            txt_id = os.path.basename(os.path.normpath(txt_id))
            with open(txt_path + txt_id + '.txt', 'r') as f:
                for line in f:
                    file.append(list(line.strip('\n').split(' ')))
                root = self.set_base(xml_path, txt_id)
                for i in range(len(file)):
                    file_xml = file[i][:]
                    self.add_xml(xml_path, txt_id, file_xml, root)
                num_box=len(file)
                if num_box==0:
                    print ('none box',txt_id)
            num_box_all+=num_box
            file = []
        print(self.name + " mission complete")
        print ('txt的数量：',num_txt)
        print ('框的数量：',num_box_all)

    def read_xml(self,in_path):
        tree = ElementTree()
        tree.parse(in_path)
        return tree

    def write_xml(self,tree, out_path):
        tree.write(out_path, encoding="utf-8", xml_declaration=True)

    def if_match(self,node, kv_map):
        for key in kv_map:
            if node.get(key) != kv_map.get(key):
                return False
        return True

    # ---------------search -----
    def find_nodes(self,tree, path):
        '''''查找某个路径匹配的所有节点
          tree: xml树
          path: 节点路径'''
        return tree.findall(path)

    def get_node_by_keyvalue(self,nodelist, kv_map):
        '''''根据属性及属性值定位符合的节点，返回节点
          nodelist: 节点列表
          kv_map: 匹配属性及属性值map'''
        result_nodes = []
        for node in nodelist:
            if if_match(node, kv_map):
                result_nodes.append(node)
        return result_nodes

    # ---------------change -----
    def change_node_properties(self,nodelist, kv_map, is_delete=False):
        '''''修改/增加 /删除 节点的属性及属性值
          nodelist: 节点列表
          kv_map:属性及属性值map'''
        for node in nodelist:
            for key in kv_map:
                if is_delete:
                    if key in node.attrib:
                        del node.attrib[key]
                else:
                    node.set(key, kv_map.get(key))

    def change_node_text(self,nodelist, text, is_add=False, is_delete=False):
        '''''改变/增加/删除一个节点的文本
          nodelist:节点列表
          text : 更新后的文本'''
        for node in nodelist:
            if is_add:
                node.text += text
            elif is_delete:
                node.text = ""
            else:
                node.text = text

    def create_node(self,tag, property_map, content):
        '''''新造一个节点
          tag:节点标签
          property_map:属性及属性值map
          content: 节点闭合标签里的文本内容
          return 新节点'''
        element = Element(tag, property_map)
        element.text = content
        return element

    def add_child_node(self,nodelist, element):
        '''''给一个节点添加子节点
          nodelist: 节点列表
          element: 子节点'''
        for node in nodelist:
            node.append(element)

    def del_node_by_tagkeyvalue(self,nodelist, tag, kv_map):
        '''''同过属性及属性值定位一个节点，并删除之
          nodelist: 父节点列表
          tag:子节点标签
          kv_map: 属性及属性值列表'''
        for parent_node in nodelist:
            children = parent_node.getchildren()
            for child in children:
                if child.tag == tag and if_match(child, kv_map):
                    parent_node.remove(child)

    def DrawObjectBox(self,Im, ObjBndBoxSet, BoxColor, save_path, txt_id):
        for ObjName, BndBoxSet in ObjBndBoxSet.items():
            for BndBox in BndBoxSet:
                cv2.rectangle(Im, (BndBox[0], BndBox[1]), (BndBox[2], BndBox[3]), BoxColor, 2)
                dsptxt = '{:s}'.format(ObjName)
                cv2.putText(Im, dsptxt, (max([(BndBox[0] + BndBox[2]) / 2 - 10, 0]), max([BndBox[3] - 3, 0])),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
                cv2.imwrite(save_path + txt_id + '.jpg', Im)

    def str2float(self,s):
        def fn(x, y):
            return x * 10 + y

        n = s.index('.')
        s1 = list(map(int, [x for x in s[:n]]))
        s2 = list(map(int, [x for x in s[n + 1:]]))
        return reduce(fn, s1) + reduce(fn, s2) / (10 ** len(s2))

    def set_base(self,save_path, file_id):
        annotation = ElementTree.Element('annotation', {})
        folder = ElementTree.SubElement(annotation, 'folder', {})
        folder.text = '15'
        filename = ElementTree.SubElement(annotation, 'filename', {})
        filename.text = file_id + '.jpg'
        source = ElementTree.SubElement(annotation, 'source', {})
        database = ElementTree.SubElement(source, 'database', {})
        database.text = 'Unknow'
        size = ElementTree.SubElement(annotation, 'size', {})
        width = ElementTree.SubElement(size, 'width', {})
        width.text = '720'
        height = ElementTree.SubElement(size, 'height', {})
        height.text = '720'
        depth = ElementTree.SubElement(size, 'depth', {})
        depth.text = '3'
        segmented = ElementTree.SubElement(annotation, 'segmented', {})
        segmented.text = '0'
        tree = ElementTree.ElementTree(annotation)
        tree.write(save_path + file_id + '.xml')
        return annotation

    def add_xml(self,save_path, file_id, file, first_element):
        # file class,bbox
        object = ElementTree.SubElement(first_element, 'object', {})
        name = ElementTree.SubElement(object, 'name', {})
        name.text = str(int(float(file[0])))
        pose = ElementTree.SubElement(object, 'pose', {})
        pose.text = 'Unspecified'
        truncated = ElementTree.SubElement(object, 'truncated', {})
        truncated.text = '0'
        difficult = ElementTree.SubElement(object, 'difficult', {})
        difficult.text = '0'
        bndbox = ElementTree.SubElement(object, 'bndbox', {})
        xmin = ElementTree.SubElement(bndbox, 'xmin', {})
        xmin.text = str(int(float(file[1])))
        ymin = ElementTree.SubElement(bndbox, 'ymin', {})
        ymin.text = str(int(float(file[2])))
        xmax = ElementTree.SubElement(bndbox, 'xmax', {})
        xmax.text = str(int(float(file[3])))
        ymax = ElementTree.SubElement(bndbox, 'ymax', {})
        ymax.text = str(int(float(file[4])))
        tree = ElementTree.ElementTree(first_element)
        tree.write(save_path + file_id + '.xml')


class xml2txt(object):
    def __init__(self):
        self.name = 'xml to txt'
    def xml_to_txt(self,xml_path, txt_path):
        dic={}
        os.chdir(xml_path)
        annotations = os.listdir('.')

        for i, file in enumerate(annotations):

            file_save = file.split('.')[0] + '.txt'
            file_txt = os.path.join(outdir, file_save)
            f_w = open(file_txt, 'w')
            in_file = open(file)
            tree = ET.parse(in_file)
            root = tree.getroot()

            for obj in root.iter('object'):
                name = obj.find('name').text
                if name not in dic.keys():
                    dic.update({name: 0})
                dic[name] += 1
                xmlbox = obj.find('bndbox')
                xn = xmlbox.find('xmin').text
                xx = xmlbox.find('xmax').text
                yn = xmlbox.find('ymin').text
                yx = xmlbox.find('ymax').text

                f_w.write(str(name)+' ')
                f_w.write(xn+' '+yn+' '+xx+' '+yx+' '+'\n')
        print(self.name + " mission complete")
        print(dic)

if __name__ == '__main__':
    indir = 'G:/git_folder/git_start/tools_pytorch/Map_calcu/txt/'
    outdir = 'G:/git_folder/git_start/tools_pytorch/Map_calcu/xml_test/'
    #a=xml2txt()
    #a.xml_to_txt(indir,outdir)
    b=txt2xml()
    b.transform(indir,outdir)