# Map_calcu
> 原代码的链接[https://github.com/Cartucho/mAP](https://github.com/Cartucho/mAP)

目前就是输入gt和predict的文件夹路径，这两种都不能用xml表示，需要用txt，格式如示例
gt:
000 xmin ymin xmax ymax

pred:
000 conf xmin ymin xmax ymax


txt_and_xml.py是用于xml和txt互相转换的，但没考虑置信度，所以在生成预测结果时最好直接得到预测txt，再用groundtruth的xml去生成对照txt