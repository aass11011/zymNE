'''
@File : t_sne.py
@Author: ZhangYiming
@Date : 2019/4/8
@Desc :  通过t-SNE 进行高维向量可视化
'''
import numpy as np
from numpy import linalg
from numpy.linalg import norm
from scipy.spatial.distance import squareform,pdist
import sklearn
from sklearn.manifold import TSNE
from sklearn.preprocessing import scale
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.manifold.t_sne import ( _joint_probabilities,_kl_divergence)
from sklearn.utils.extmath import *
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import matplotlib
import seaborn as sns
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook" , font_scale=1.5 , rc={"lines.linewidth":2.5})
class tSNE:
    def __init__(self,embeddings):
        self.embeddings = embeddings
        self.RS = 20190410
        self.res_vec = []
        self.train()


    def train(self):
        random_state = self.RS
        self.res_vec = TSNE(random_state=random_state).fit_transform(self.embeddings)

