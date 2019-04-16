'''
@File : gml_util.py
@Author: ZhangYiming
@Date : 2019/4/10
@Desc : 将gml文件处理为可学习的边关系格式文件
'''
import networkx as nx

path = "E://PYTHONGRAM//zymNE//data//{}//{}.gml"
filename = "karate"
g = nx.read_gml(path.format(filename,filename),label=None)
print("done")