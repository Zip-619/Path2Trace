from convert import *

"""
最后的目标是
#node
#type
#edge list
offsets list
edges list
weights list

需要两个index表，nodes和type
"""


def is_numbers_only(nodes):
    try:
        list(map(int, nodes))
    except:
        return False
    return True


def load(input, sep=' '):
    """
    加载数据，格式：type f_node t_node weight

    """
    nodes = set()
    titles = set()
    with open(input, 'r') as inf:
        for line in inf:
            if line.startswith('#') or line.startswith('%'):
                continue
            arr = line.strip().split(sep)
            titles.add(arr[0])
            nodes.add(arr[1])
            nodes.add(arr[2])
    number_of_nodes = len(nodes)
    number_of_titles = len(titles)
    isnumbers_nodes = is_numbers_only(nodes)
    isnumbers_titles = is_numbers_only(titles)
    print('Node IDs are numbers : %s, Titles IDs are numbers : %s.' % (isnumbers_nodes, isnumbers_titles))
    if isnumbers_nodes:
        node2id = dict(zip(sorted(map(int, nodes)), range(number_of_nodes)))  # k对应的是节点，value对应的是index
    else:
        node2id = dict(zip(sorted(nodes), range(number_of_nodes)))
    if isnumbers_titles:
        title2id = dict(zip(sorted(map(int, titles)), range(number_of_titles)))  # k对应的是title，value对应的是index
    graphs = dict()
    with open(input, 'r') as inf:
        for line in inf:
            if line.startswith('#') or line.startswith('%'):
                continue
            arr = line.strip().split(sep)
            title = title2id[int(arr[0])] if isnumbers_titles else title2id[arr[0]]
            src = node2id[int(arr[1])] if isnumbers_nodes else node2id[arr[1]]
            tgt = node2id[int(arr[2])] if isnumbers_nodes else node2id[arr[2]]
            weight = float(arr[3])
            if title not in graphs:
                graphs[title] = defaultdict(set)
            graphs[title][src].add((tgt, weight))
    indptrs = list()
    indices = list()
    weights = list()
    for title in range(number_of_titles):
        indptr = np.zeros(number_of_nodes + 1, dtype=np.int32)
        indptr[0] = 0
        for i in range(number_of_nodes):
            indptr[i + 1] = indptr[i] + len(graphs[title][i])
        number_of_edges = indptr[-1]
        indice = np.zeros(number_of_edges, dtype=np.int32)
        weight = np.zeros(number_of_edges, dtype=np.float32)
        cur = 0
        for node in range(number_of_nodes):
            for adjv in sorted(graphs[title][node]):
                indice[cur], weight[cur] = adjv
                cur += 1
        indices.append(indice)
        weights.append(weight)
        indptrs.append(indptr[:-1])
    return indptrs, indices, weights


def xgfs2file(outf, indptrs, indices, weights):
    nv = indptrs[0].size
    nt = len(indptrs)
    MAGIC = 'XGFS'.encode('utf8')
    outf.write(MAGIC)
    outf.write(pack('q', nv))
    outf.write(pack('q', nt))
    ne = np.zeros(nt, dtype=np.int32)
    for i in range(nt):
        ne[i] = indices[i].size
    outf.write(pack('%di' % nt, *ne))
    for i in range(nt):
        # print(indptrs[i])
        outf.write(pack('%di' % nv, *indptrs[i]))
    print('############')
    for i in range(nt):
        # print(indices[i])
        outf.write(pack('%di' % ne[i], *indices[i]))
    print('############')
    for i in range(nt):
        # print(weights[i])
        outf.write(pack('%df' % ne[i], *weights[i]))



def process(input, output):
    indptrs, indices, weights = load(input)
    with open(output, 'wb') as fout:
        xgfs2file(fout, indptrs, indices, weights)


if __name__ == '__main__':
    # a, b, c = load('data/tmp.txt')
    # for i in range(len(a)):
    #     print('******')
    #     print(a[i])
    #     print(b[i])
    #     print(c[i])
    process('all_T.train', 'src/all_T.bcsr')
