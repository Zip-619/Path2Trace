#!/usr/bin/python
# -*- coding: UTF-8 -*-

from sklearn.preprocessing import scale

import hierarch_network as hn
from subspace import SpaceNE
from tree import *
from utils import *
from walker import *
import pickle
import numpy as np

# os.environ["CUDA_VISIBLE_DEVICES"] = '0'

basevectors = dict()
node_embedding = dict()


def train(root, input_dim, init_embeddings):
    print("current root:", root)
    if len(tree[root].childst) == 0:
        return

    flag = False
    for i in tree[root].childst:
        if i < leafnode_num:
            flag = True
            break
    if flag:
        print("bottom!")
        return

    global node_embedding
    X, node_embedding = hn.build_X(root, tree, node_embedding, hi_network, init_embeddings)

    if root == params["root"]:
        X_scaled = list()
        for i in range(len(X)):
            X_scaled.append(100 * scale(X[i]))
    else:
        X_scaled = X
    se = SpaceNE(X_scaled, len(tree[root].childst), input_dim, params)
    se.train()
    W = se.load_W()
    print(tree[root].childst)

    for (index, i) in enumerate(tree[root].childst):
        basevectors[i] = orth(W[index])  # gram_schmidt
        projection = np.dot(basevectors[i].transpose(), X[index].transpose()).transpose().astype('float32') 
        node_embedding[i] = embed_array_to_dict(hi_network[i], projection)
        train(i, input_dim - 1, init_embeddings)  # here we constrait the max_dim to (d - 1)


if __name__ == '__main__':
    params = load_json_file("../conf/case_travel.json")

    tree, n, leafnode_num = extract_hierarchy(params["base_path"] + params["path_tree"])
    hi_network = hn.build_hierach_network(tree, leafnode_num)

    init_embeddings = Walks(params).get_embeddings()

    train(params["root"], params["SpaceNE"]["dimension"], init_embeddings)
    res_X = reconstruction_X_top(basevectors, node_embedding, tree, leafnode_num)

    save_res(basevectors, node_embedding, tree, res_X, params)  # save embeddings and other parameters
    # save_resX_txt(res_X, "embeddings.txt") # only save embeddings


    with open('../../../data/pengpai/labeled_data/idx_address_network_no_edgedata/nodeidxmaps.pickle', 'rb') as file:
        addr2idxmaps = pickle.load(file)
        file.close()
    addr2idx = addr2idxmaps['poi']
    idx2addr = {val: key for key, val in addr2idx.items()}

    emb = np.load('data/output/node_embeddings.npy')

    addr2emb = {}

    for i, line in enumerate(emb):
        poi_idx = i

        addr2emb[idx2addr[poi_idx]] = np.array(line).astype(np.float32)

    with open('../../../represent_learning/poi_embeddings/spacene_poi_embedding.pickle', 'wb') as file:
        pickle.dump(obj=addr2emb, file=file)
        file.close()
