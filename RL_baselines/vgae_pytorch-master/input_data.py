'''
****************NOTE*****************
CREDITS : Thomas Kipf
since datasets are the same as those in kipf's implementation, 
Their preprocessing source was used as-is.
*************************************
'''
import pickle as pkl
import sys

import networkx as nx
import numpy as np
import scipy.sparse as sp


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def load_travel_data():
    g = nx.read_edgelist('../poi_net_edgelist_intnode.txt',nodetype = int)
    adj = nx.to_scipy_sparse_matrix(g)
    feature_arr = np.loadtxt('../poi_feature.txt')
    feature = sp.lil_matrix(feature_arr)
    return adj,feature