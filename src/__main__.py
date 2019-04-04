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


def main():
    t1 = time.time()
    g = graph.Graph()
    print("Reading...")
    input = "E://PYTHONGRAM//zymNE//data//blogCatalog//bc_adjlist.txt"
    graph_format = "adjlist"
    encoder_layer_list = ast.literal_eval('[1000, 128]')
    if graph_format == 'adjlist':
        g.read_adjlist(filename=input)
    elif graph_format == 'edgelist':
        g.read_edgelist(filename=input, weighted=False,
                        directed=False)
    # a list of numbers of the neuron at each encoder layer, the last number is the dimension of the output node representation
    # 每层encoder神经元的数量列表，最后一个数字是输出节点向量的维度
    alpha = 1e-6
    # alhpa is a hyperparameter in SDNE
    beta = 5.0
    # beta is a hyperparameter in SDNE
    nu1 = 1e-5
    # nu1 is a hyperparameter in SDNE
    nu2 = 0.0001
    # nu2 is a hyperparameter in SDNE
    bs = 200
    # batch size of SDNE'
    epochs = 5
    # The training epochs of LINE and GCN
    lr = 0.01
    # learning rate
    output = "E://PYTHONGRAM//zymNE//result//res_vec.txt"
    model = sdne.SDNE(g, encoder_layer_list=encoder_layer_list,
                      alpha=alpha, beta=beta, nu1=nu1, nu2=nu2,
                      batch_size=bs, epoch=epochs, learning_rate=lr)
    model.save_embeddings(output)
if __name__ == "__main__":
    random.seed(32)
    np.random.seed(32)
    main()