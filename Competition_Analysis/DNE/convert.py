#   encoding: utf8
#   convert.py
"""Converter for three common graph formats (MATLAB sparse matrix, adjacency
list, edge list) can be found in the root directory of the project.
"""

import logging
from collections import defaultdict
from struct import pack

import numpy as np
from scipy.io import loadmat

MAGIC = 'XGFS'.encode('utf8')


def mat2csr(filename, varname):
    mat = loadmat(filename)[varname].tocsr()
    return mat.indptr[:-1], mat.indices, mat.data  # we treat weights in xgfs2file


def xgfs2file(outf, indptr, indices, weights=None):
    nv = indptr.size
    ne = indices.size
    logging.info('num vertices=%d; num edges=%d;', nv, ne)
    assert np.all(weights >= 0), 'negative weights are not allowed'
    outf.write(MAGIC)
    outf.write(pack('q', nv))
    outf.write(pack('q', ne))
    outf.write(pack('%di' % nv, *indptr))
    outf.write(pack('%di' % ne, *indices))
    if weights is not None:
        if not np.all(weights == 1):
            outf.write(pack('%df' % ne, *weights))


def is_numbers_only(nodes):
    try:
        list(map(int, nodes))
    except:
        return False
    return True


def list2mat(input, undirected, sep, format):
    nodes = set()
    with open(input, 'r') as inf:
        for line in inf:
            if line.startswith('#') or line.startswith('%'):
                continue
            line = line.strip()
            splt = line.split(sep)
            if not splt: continue
            if format == 'edgelist':
                assert len(splt) == 2, 'In edgelist there should be 2 values per line '
            if format == 'weighted_edgelist':
                assert len(splt) == 3, 'In weighted edgelist there should be 3 values per line'
                splt = splt[:-1] # 去掉weight 以获得所有的node
            for node in splt:
                nodes.add(node)
    number_of_nodes = len(nodes)
    isnumbers = is_numbers_only(nodes)
    logging.info('Node IDs are numbers: %s', isnumbers)
    if isnumbers:
        node2id = dict(zip(sorted(map(int, nodes)), range(number_of_nodes))) # k对应的是节点，value对应的是index
    else:
        node2id = dict(zip(sorted(nodes), range(number_of_nodes)))
    graph = defaultdict(set) # 存储的key是一个节点index, value是一个set，其中每个item是(target_node, weight) # 邻接表
    with open(input, 'r') as inf:
        for line in inf:
            if line.startswith('#') or line.startswith('%'):
                continue
            line = line.strip()
            splt = line.split(sep)
            if not splt: continue
            weight = 1
            src = node2id[int(splt[0])] if isnumbers else node2id[splt[0]]
            if format == "weighted_edgelist":
                weight = float(splt[-1]) # 强制类型转换，可能会出现负数
                # assert (weight >= 0), splt
                splt = splt[:-1]
            for node in splt[1:]:
                if isnumbers:
                    tgt = node2id[int(node)]
                else:
                    tgt = node2id[node]
                graph[src].add((tgt, weight))
                if undirected:
                    graph[tgt].add((src, weight))
    indptr = np.zeros(number_of_nodes + 1, dtype=np.int32)
    indptr[0] = 0
    for i in range(number_of_nodes):
        indptr[i + 1] = indptr[i] + len(graph[i])
    number_of_edges = indptr[-1]
    indices = np.zeros(number_of_edges, dtype=np.int32)
    weights = np.zeros(number_of_edges, dtype=np.float32)
    cur = 0
    for node in range(number_of_nodes):
        for adjv in sorted(graph[node]):
            indices[cur], weights[cur] = adjv
            cur += 1
    # assert np.all(weights > 0), 'xxxxxxxxxxx negative weights are not allowed'
    return indptr[:-1], indices, weights


def process(format, matfile_variable_name, undirected, sep, input, output):
    if format == "mat":
        indptr, indices, weights = mat2csr(input, matfile_variable_name)
    elif format in ['weighted_edgelist', 'edgelist', 'adjlist']:
        indptr, indices, weights = list2mat(input, undirected, sep, format)

    with open(output, 'wb') as fout:
        xgfs2file(fout, indptr, indices, weights)

def main(format, matfile_variable_name, undirected, sep, input, output):
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                        level=logging.INFO)
    logging.info('convert graph from %s to %s', input, output)
    process(format, matfile_variable_name, undirected, sep, input, output)
    logging.info('done.')


if __name__ == "__main__":
    format = 'weighted_edgelist'
    matfile_variable_name = 'network'
    undirected = False  # False的时候表示图为有向图
    sep = ' '
    input = 'data/test.txt'
    output = 'test.bcsr'
    exit(main(format, matfile_variable_name, undirected, sep, input, output))
