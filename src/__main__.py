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


def main():
    t1 = time.time()
    g = graph.Graph()
    print("Reading...")
    #input = "..//data//blogCatalog//bc_adjlist.txt"
    #input = "..//data//football//football.gml"
    #input = "..//data//karate//karate.gml"
    input = "..//data//107//107.adj"
    graph_format = "adjlist"
    encoder_layer_list = ast.literal_eval('[1000, 128]')
    if graph_format == 'adjlist':
        g.read_adjlist(filename=input)
    elif graph_format == 'edgelist':
        g.read_edgelist(filename=input, weighted=False,
                        directed=False)
    elif graph_format == 'gml':
        g.read_gml(filename=input)
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
    weakening_rate = [0.5,1.0,2.0,3.5,4.5,5.0]

    #step 1  network embedding
    model.save_embeddings(path.format(res_vec_name))

    comms_num = 5
    #step 2 kmeans cluster
    model.k_means(comms_num)
    model.save_centroids(path.format(centroids_name))
    model.save_comms(path.format(output_cluster.format('0')))
    #step 3 visualize
    tsne = t_sne.tSNE(model.embeddings,model.centroids,model.clusterAssment[:,0],model.g.look_back_list)
    png_name = "original_karate_cluster.png"
    tsne.scatter(path,png_name)

    #step 4 offcenter and weaken comms

    for wr in weakening_rate:
        print("-----weaken rate:{}".format(wr))
        model.weaken_community(wr)
        model.k_means_keep_center(comms_num)
        model.save_comms(path.format(output_cluster.format(str(wr).replace('.', '_'))),type="offcenter")
        # step 5 visualize
        tsne = t_sne.tSNE(model.embeddings,model.centroids,model.clusterAssment[:,0],model.g.look_back_list)
        png_name = "weaken_karate_cluster_{}.png".format(str(wr).replace('.', '_'))
        tsne.scatter(path,png_name)

        model.communities_detection(path,wr)






if __name__ == "__main__":
    random.seed(32)
    np.random.seed(32)
    main()