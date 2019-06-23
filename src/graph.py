'''
@File : graph.py
@Author: ZhangYiming
@Date : 2019/3/25
@Desc :
'''
import networkx as nx
import pickle as pkl
import numpy as np
import scipy.sparse as sp

class Graph(object):
    def __init__(self):
        self.G = None
        self.look_up_dict = {}
        self.look_back_list = []
        self.node_size = 0

    def encode_node(self):
        look_up = self.look_up_dict
        look_back = self.look_back_list
        for node in self.G.nodes():
            look_up[node] = self.node_size   #look_up_dict  k: node v:读入顺序
            look_back.append(node)             #look_back_list 节点的读入顺序list
            self.node_size += 1
            self.G.nodes[node]['status'] = ''

    def read_g(self, g):
        self.G = g
        self.encode_node()

    def read_adjlist(self,filename):
        self.G = nx.read_adjlist(filename, create_using=nx.DiGraph())
        for i, j in self.G.edges():
            self.G[i][j]['weight'] = 1.0
        self.encode_node()

    def read_edgelist(self,filename,weighted=False,directed=False):
        self.G = nx.DiGraph()

        if directed:
            def read_unweighted(l):
                src , dst = l.split()
                self.G.add_edge(src , dst)
                self.G[src][dst]['weight'] = 1.0
            def read_weight(l):
                src,dst,w = l.split()
                self.G.add_edge(src, dst)
                self.G[src][dst]['weight'] = float(w)
        else:
            def read_unweighted(l):
                src ,dst = l.split()
                self.G.add_edge(src,dst)
                self.G.add_edge(dst,src)
                self.G[src][dst]['weight'] = 1.0
                self.G[dst][src]['weight'] = 1.0
            def read_weighted(l):
                src, dst, w = l.split()
                self.G.add_edge(src, dst)
                self.G.add_edge(dst, src)
                self.G[src][dst]['weight'] = float(w)
                self.G[dst][src]['weight'] = float(w)
        fin = open(filename, 'r')
        func = read_unweighted
        if weighted:
            func = read_weighted
        while 1:
            l = fin.readline()
            if l == '':
                break
            func(l)
        fin.close()
        self.encode_node()

    def read_gml(self,filename):
        self.G =nx.read_gml(filename)
        for i, j in self.G.edges():
            self.G[i][j]['weight'] = 1.0
        self.encode_node()
    def read_csv(self,filename):
        self.G = nx.DiGraph()
        fin = open(filename,'r')
        for l in fin.readlines():
            if l=='':
                break
            src,dst = l.strip().split(',')
            self.G.add_edge(src, dst)
            self.G.add_edge(dst, src)
            self.G[src][dst]['weight'] = 1.0
            self.G[dst][src]['weight'] = 1.0
        self.encode_node()
    def read_node_label(self, filename):
        fin = open(filename, 'r')
        while 1:
            l = fin.readline()
            if l == '':
                break
            vec = l.split()
            self.G.nodes[vec[0]]['label'] = vec[1:]
        fin.close()

    def read_node_features(self, filename):
        fin = open(filename, 'r')
        for l in fin.readlines():
            vec = l.split()
            self.G.nodes[vec[0]]['feature'] = np.array(
                [float(x) for x in vec[1:]])
        fin.close()

    def read_node_status(self, filename):
        fin = open(filename, 'r')
        while 1:
            l = fin.readline()
            if l == '':
                break
            vec = l.split()
            self.G.nodes[vec[0]]['status'] = vec[1]  # train test valid
        fin.close()


    def read_edge_label(self, filename):
        fin = open(filename, 'r')
        while 1:
            l = fin.readline()
            if l == '':
                break
            vec = l.split()
            self.G[vec[0]][vec[1]]['label'] = vec[2:]
        fin.close()

    def read_groundtruth(self,filename):
        # groundtruth  k: node v:true_label
        groundtruth = {}
        fin = open(filename,'r')
        label = 0
        while 1:
            l = fin.readline()
            if l == '':
                break
            vec = l
            for i in vec:
                groundtruth.update({i:label})
            label += 1
        return  groundtruth