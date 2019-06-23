'''
@File : test.py
@Author: ZhangYiming
@Date : 2019/6/10
@Desc :
'''
trues = [0,0,0,1,1,1,2,2,2,1,1]
pred =  [1,1,1,1,2,2,3,3,3,3,1]

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

trues,pred = align_truth_pred(trues,pred)
print("trues:",trues)
print("pred:",pred)