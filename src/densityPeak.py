'''
@File : densityPeak.py
@Author: ZhangYiming
@Date : 2019/5/31
@Desc : 密度峰值
'''
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as ds
import matplotlib.colors
'''
定义聚类中心距离
1、每个点的密度从大到小排列：pi>pj>pk ;密度最大的点的聚类中心距离与其他点的聚类中心距离确定方法是不一样的
2、先确定密度最大的点的聚类中心距离 i点的聚类中心距离是与i点最远的点j到i的距离
3、再确定其他点的聚类中心距离，其他的点的聚类中心距离等于在比自己密度大的点的集合中，与该点距离最小的那个距离
4、依次确定所有的聚类中心距离
'''
def get_point_density(datas,labers,min_distance,points_number):
    data = datas.tolist()
    laber = labers.tolist()
    distance_all = np.random.rand(points_number,points_number)
    point_density = np.random.rand(points_number)
    #计算各点之间的距离
    for i in range(points_number):
        for j in  range(points_number):
            pass
            '''
            计算距离
            '''
    #计算各点的密度
    for i in range(points_number):
        density = 0
        for j in range(points_number):
            if distance_all[i][j] > 0 and distance_all[i][j] < min_distance:
                density = density+1
            point_density[i] = density
    return distance_all, point_density

#计算密度最大的点的聚类中心距离
def get_max_distance(distance_all,point_density,laber):
    point_density = point_density.tolist()
    max_point_num = int(max(point_density))

    return distance_all[point_density.index(max_point_num)]

#计算得到各点的聚类中心距离
def get_each_distance(distance_all,point_density,data,laber):
    nn = []
    for i in range(len(point_density)):
        aa = []
        for j in range(len(point_density)):
            if point_density[i] < point_density[j]:
                aa.append(j)
        ll = get_min_distance(aa,i,distance_all,point_density,laber)
        nn.append(ll)
    return nn

#到点密度大于自身的最近点的距离
def get_min_distance(aa,i,distance_all,point_density,data,laber):
    min_distance = []
    if aa != []:
        for k in aa:
            min_distance.append(distance_all[i][k])
            return min(min_distance)