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
    def __init__(self,embeddings,centroids,colors,labels):
        self.vector = embeddings
        self.RS = 20190410
        self.res_vec = []
        self.centers = centroids
        self.train()
        self.labels = labels
        self.colors = colors

    def train(self):
        embeddings = self.vector
        random_state = self.RS
        self.res_vec = TSNE(random_state=random_state).fit_transform(np.vstack((self.centers,embeddings)))
        #self.centers = TSNE(random_state=random_state).fit_transform(self.centers)
        print("res_vec :{}".format(self.res_vec))
        #print("centers :{}".format(self.centers))

    def scatter(self,path,filename):
        # We choose a color palette with seaborn
        palette = np.array(sns.color_palette("hls", 10))
        f = plt.figure(figsize=(8, 8))
        ax = plt.subplot(aspect='equal')


        self.colors = np.array(self.colors).reshape(-1)
        k = self.centers.shape[0]
        ax.scatter(self.res_vec[0:k, 0], self.res_vec[0:k, 1], lw=0, s=40,
                   c=range(k))
        sc = ax.scatter(self.res_vec[k:, 0], self.res_vec[k:, 1], lw=0, s=40,
                        c=palette[self.colors])

        plt.xlim(-25, 25)
        plt.ylim(-25, 25)
        ax.axis('off')
        ax.axis('tight')

        '''
        # 为每个节点加标签

        
        for i in range(len(self.res_vec[k:,0])):
            txt = ax.text(self.res_vec[k+i,0], self.res_vec[k+i,1], str(self.labels[i]), fontsize=8)

            txt.set_path_effects([
                PathEffects.Stroke(linewidth=5, foreground="w"),
                PathEffects.Normal()
            ])
        '''
        plt.savefig(path.format(filename))