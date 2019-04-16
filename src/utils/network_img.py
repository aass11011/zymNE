'''
@File : network_img.py
@Author: ZhangYiming
@Date : 2019/4/12
@Desc : 绘制网络关系图
'''
import networkx as nx
import matplotlib.pyplot as plt

path = "E://PYTHONGRAM//zymNE//data//football//{}"
filename = "football.gml"
g = nx.read_gml(path.format(filename))
'''
node_size
node_color
node_shape
alpha
width
edge_color
style
with_lables
font_size
font_color
'''

'''
circular_layout:在圆环上均匀分布
random_layout
shell_layout
spring_layout
spectral_layout:根据图的拉普拉斯特征向量排列节点
'''
plt.show()
