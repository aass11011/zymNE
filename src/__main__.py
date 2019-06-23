'''
@File : __main__.py
@Author: ZhangYiming
@Date : 2019/3/25
@Desc :
'''
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import ast
import numpy as np
import random
import time
from src import sdne
from src import graph
from src import t_sne
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import json
from sklearn.metrics.cluster import normalized_mutual_info_score



def find_index(array,x):
    indexs = list()
    for index,i in enumerate(array):
        if i==x:
            indexs+=[index]
    return indexs



def align_truth_pred(trues,pred):
    j_collection = set()
    i_collection = set()
    for i_index,i in enumerate(trues):
        i_indexs = find_index(trues,i)
        if i_index in i_collection:
            continue
        for j_index,j in enumerate(pred):
            if j_index in j_collection:
                continue
            else:
                j_indexs = find_index(pred,j)
                for x in j_indexs:
                    pred[x] = i
                for x in j_indexs:
                    j_collection.add(x)
                break
        for x in i_indexs:
            i_collection.add(x)
    return trues,pred




def main():
    t1 = time.time()
    g = graph.Graph()
    print("Reading...")
    #input = "..//data//blogCatalog//bc_adjlist.txt"
    #input = "..//data//football//football.gml"
    #input = "..//data//karate//karate.gml"
    #input = "../data/food/food_edges.csv"
    input = "..//data//amazon//com-amazon.ungraph.txt"
    graph_format = "edgelist"
    encoder_layer_list = ast.literal_eval('[1000, 128]') #[1000,128]
    if graph_format == 'adjlist':
        g.read_adjlist(filename=input)
    elif graph_format == 'edgelist':
        g.read_edgelist(filename=input, weighted=False,
                        directed=False)
    elif graph_format == 'gml':
        g.read_gml(filename=input)
    elif graph_format == "json":
        g.read_csv(filename = input)
    nodelabels = open("../data/amazon/com-amazon.all.dedup.cmty.txt").readlines()
    groundpath = "../data/amazon/com-amazon.all.dedup.cmty.txt"
    '''
    groundtruth = {}
    for i in nodelabels:
        nodelabel = i.strip().split(" ")
        node,label = nodelabel[0],nodelabel[1]
        groundtruth.update({node:label})
    '''
    groundtruth = g.read_groundtruth(groundpath)

    # a list of numbers of the neuron at each encoder layer, the last number is the dimension of the output node representation
    # 每层encoder神经元的数量列表，最后一个数字是输出节点向量的维度
    alpha = 1e-6
    # alhpa is a hyperparameter in SDNE
    beta = 5.0
    # beta is a hyperparameter in SDNE
    nu1 = 1e-5
    # nu1 is a hyperparameter in SDNE
    nu2 = 0.0001
    # nu2 ispi a hyperparameter in SDNE
    bs = 200
    # batch size of SDNE'
    epochs = 5
    # The training epochs of LINE and GCN
    lr = 0.01
    # learning rate
    path = "..//result//{}"
    res_vec_name = "res_vec.txt"
    centroids_name = "centroids.txt"
    output_cluster = "res_{}_comms.txt"
    model = sdne.SDNE(g, encoder_layer_list=encoder_layer_list,
                      alpha=alpha, beta=beta, nu1=nu1, nu2=nu2,
                      batch_size=bs, epoch=epochs, learning_rate=lr)


    #step 1  network embedding
    model.save_embeddings(path.format(res_vec_name))

    # elbow rule method

    '''
    
    n_clusters = 80
    wcss = []
    #step 2 kmeans cluster
    #y_pred = DBSCAN(eps=1.5,min_samples=5).fit_predict(model.embeddings)
    for i in range(1,n_clusters):
        y_pred = KMeans(n_clusters=i,init="k-means++", random_state=1231)\
            .fit(model.embeddings)

        wcss.append(y_pred.inertia_)

    plt.plot(range(1,n_clusters),wcss)
    plt.title('肘部方法')
    plt.xlabel('聚类数量')
    plt.ylabel('wcss')
    # plt.ion()
    plt.savefig("../result/elbowrule")
    plt.show()
    '''

    
    labels = []
    # 将groundtruth中的节点顺序重排，使与embedding中的一致
    for i, embedding in enumerate(g.look_back_list):
        labels.append(groundtruth.get(g.look_back_list[i]))
    
    n = len(set(labels))
    kmeans_model = KMeans(n_clusters=n, init="k-means++", random_state=1231)
    y_pred = kmeans_model.fit(model.embeddings).labels_.tolist()

    #y_pred = list(y_pred)
    #step 3 visualize
    tsne = t_sne.tSNE(model.embeddings)




    plt.scatter(tsne.res_vec[:, 0], tsne.res_vec[:, 1],c=y_pred)
    plt.savefig("../result/res.png")
    #step 4 caculate NMI
    #groundtruth,y_pred = align_truth_pred(groundtruth,y_pred)
    score = normalized_mutual_info_score(labels,y_pred)
    print("NMI score",score)
    fo = open("../result/metrics","w")
    fo.write("NMI score :{}".format(score))





if __name__ == "__main__":
    random.seed(32)
    np.random.seed(32)
    main()



