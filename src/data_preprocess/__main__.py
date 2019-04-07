'''
@File : __main__.py
@Author: ZhangYiming
@Contact : 381766867@qq.com 
@Date : 2019/4/6
@Desc :
'''
from src.data_preprocess import edge2adj_list
if __name__ =='__main__':
    path = "E://PYGRAM//zymNE//data//107//107.edges"
    graph = edge2adj_list.ngraph()
    graph.read_edgelist(filename=path)
    save_path = "E://PYGRAM//zymNE//data//107//107.adj"
    graph.save_adjlist(save_path)