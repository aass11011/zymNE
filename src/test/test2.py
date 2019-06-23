'''
@File : test2.py
@Author: ZhangYiming
@Date : 2019/6/11
@Desc :
'''
import numpy as np
from sklearn.metrics import normalized_mutual_info_score
'''

A = np.array([[1,1,1],
             [2,2,2]])
B = np.array([[3,3,3],
              [4,4,4]])
print(np.hstack((A,B))) # vertical stack,属于一种上下合并,即对括号中的两个整体进行对应操作
'''
A = [[1,2,3],[4,5,6],[7,8,9]]
B = [[1,4,5],[2,3,6],[7,8,9]]
score = normalized_mutual_info_score(A,B)
print(score)