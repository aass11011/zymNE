'''
@File : edge2adj_list.py
@Author: ZhangYiming
@Contact : 381766867@qq.com 
@Date : 2019/4/6
@Desc : 三元组（二元组）边关系 转换成 邻接矩阵向量
'''
import networkx as nx
import numpy as np

class ngraph:
    def __init__(self):
        self.G = None

    def read_edgelist(self,filename,weighted = False,directed=False):
        self.G = nx.DiGraph()
        if directed:
            def read_unweighted(l):
                src,dst = l.split()
                self.G.add_edge(src ,dst)
                self.G[src][dst]['weight'] = 1.0

            def read_weighted(l):
                src,dst,w = l.split()
                self.G.add_edge(src,dst)
                self.G[src][dst]['weight'] = float(w)
        else:
            def read_unweighted(l):
                src, dst = l.split()
                self.G.add_edge(src,dst)
                self.G.add_edge(dst,src)
                self.G[src][dst]['weight'] = 1.0
                self.G[dst][src]['weight'] = 1.0
            def read_weighted(l):
                src,dst,w = l.split()
                self.G.add_edge(src, dst)
                self.G.add_edge(dst, src)
                self.G[src][dst]['weight'] = float(w)
                self.G[dst][src]['weight'] = float(w)
        fin = open(filename,'r')
        func = read_unweighted
        if weighted:
            func = read_weighted
        while 1:
            l = fin.readline()
            if l =='':
                break
            func(l)
        fin.close()

    def save_adjlist(self,filename):
        fout = open(filename,'w')
        for (node,node_adj) in self.G.adj.items():
            fout.write("{} {}\n".format(node, ' '.join(str(k) for (k,v) in node_adj.items())))
