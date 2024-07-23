import random
from typing import Dict

import numpy as np
from sklearn.metrics import roc_auc_score


class Embedding(object):
    def __init__(self, embedding_path: str, dimensions: int, index_uri: dict = None):
        self.dimensions = dimensions
        self.embeddings = self.load_embeddings(embedding_path)
        if self.embeddings.shape[0] != len(index_uri):
            print('Embedding 的节点数不等于字典的长度')
        self.index: Dict[str, int] = {}
        if index_uri:
            # self.load_index(index_uri) # 从文件中获取
            for k in index_uri:
                self.index[str(k)] = index_uri[k]  # 直接从列表中获取
        print(f"Done loading {len(self.index)} items.")

    def load_embeddings(self, file_name: str) -> np.ndarray:
        print("  embeddings...")
        embeddings = np.fromfile(file_name, dtype=np.float32)
        length = embeddings.shape[0]
        assert length % self.dimensions == 0, f"The number of floats ({length}) in the embeddings is not divisible by" \
                                              f"the number of dimensions ({self.dimensions})!"
        embedding_shape = [int(length / self.dimensions), self.dimensions]
        embeddings = embeddings.reshape(embedding_shape)
        print(f"Done loading embeddings (shape: {embeddings.shape}).")
        return embeddings

    def load_index(self, index_path: str) -> None:
        print("Loading uri index...")
        with open(index_path, "r") as file:
            for line in [line.strip() for line in file.readlines()]:
                index, uri = line.split(",", 1)
                self.index[uri] = int(index)
        print(f"Done loading {len(self.index)} items.")

    def __getitem__(self, item) -> np.ndarray:
        if self.index and isinstance(item, str):
            return self.embeddings[self.index[item]]
        return self.embeddings[item]


def verify(s_vec, t_vec, testPath):
    score = []
    label = []
    count = 0
    max_v, min_v = get_max_min(s_vec, t_vec)
    for line in open(testPath):
        count += 1
        f = int(line.strip().split(' ')[0])
        t = int(line.strip().split(' ')[1])
        l = int(line.strip().split(' ')[2])
        # if f not in s_vec or t not in t_vec: # 直接过滤未见过的节点
        #     continue
        try:
            s = np.dot(s_vec[f], t_vec[t])
        except KeyError:  # 对于没见过的节点，直接在最大值和最小值之间随机选择一个
            s = random.sample([min_v, max_v], 1)[0]
        score.append(s)
        label.append(l)
    auc = roc_auc_score(label, score)
    print(auc)
    return auc


def get_max_min(s_vec, t_vec):
    s_list = list()
    t_list = list()
    for node in s_vec:
        s_list.append(s_vec[node])
        t_list.append(t_vec[node])
    s_list = np.asarray(s_list)
    t_list = np.asarray(t_list)
    res = np.matmul(s_list, t_list.T)
    max_v = np.max(res)
    min_v = np.min(res)
    return max_v, min_v


def get_max_min_for_multi(s_vec, t_vec, l_emb):
    """
    为多任务学习计算最大值最小值
    :param s_vec: s向量 字典
    :param t_vec: t向量 字典
    :param l_emb: 某个特定岗位表征
    :return:
    """
    s_list = list()
    t_list = list()
    for node in s_vec:
        s_list.append(s_vec[node])
        t_list.append(t_vec[node])
    s_list = np.asarray(s_list)
    t_list = np.asarray(t_list)
    res = np.matmul(np.multiply(s_list, l_emb), t_list.T)
    max_v = np.max(res)
    min_v = np.min(res)
    return max_v, min_v


# final_embedding[node] = np.append(first_order_embedding[node], second_order_embedding[node])
def verify_G_GT(s_vec, t_vec, T_s_vec, T_t_vec, testPath):
    score = []
    label = []
    final_S = dict()
    final_T = dict()
    for node in s_vec:
        final_S[node] = np.append(s_vec[node], T_t_vec[node])
        final_T[node] = np.append(t_vec[node], T_s_vec[node])

    max_v, min_v = get_max_min(final_S, final_T)
    for line in open(testPath):
        f = int(line.strip().split(' ')[0])
        t = int(line.strip().split(' ')[1])
        l = int(line.strip().split(' ')[2])
        # if f not in s_vec or t not in t_vec or f not in T_t_vec or t not in T_s_vec:
        #     continue
        try:
            s = np.dot(final_S[f], final_T[t])
        except KeyError:  # 对于未见过的节点
            s = random.sample([min_v, max_v], 1)[0]
        score.append(s)
        label.append(l)
    auc = roc_auc_score(label, score)
    print(auc)
    return auc


# 针对多任务的验证
def verify_multi(s_vec, t_vec, l_vec, testPath):
    score = []
    label = []

    # 首先是获得测试集的网络名称
    for line in open(testPath):
        arr = line.strip().split(' ')
        if len(arr) == 3:
            try:
                layer = int(testPath.split('.test')[0].split('/')[-1])
            except ValueError:
                layer = int(testPath.split('.test')[
                            0].split('/')[-1].split('_')[-1])
        else:
            layer = int(line.strip().split(' ')[0])
        break
    max_v, min_v = get_max_min_for_multi(s_vec, t_vec, l_vec[layer])
    for line in open(testPath):
        arr = line.strip().split(' ')
        if len(arr) == 3:
            try:
                layer = int(testPath.split('.test')[0].split('/')[-1])
            except ValueError:
                layer = int(testPath.split('.test')[
                            0].split('/')[-1].split('_')[-1])
            f = int(line.strip().split(' ')[0])
            t = int(line.strip().split(' ')[1])
            l = int(line.strip().split(' ')[2])
        else:
            layer = int(line.strip().split(' ')[0])
            f = int(line.strip().split(' ')[1])
            t = int(line.strip().split(' ')[2])
            l = int(line.strip().split(' ')[3])
        # if f not in s_vec or t not in t_vec:
        #     continue
        try:
            c = np.multiply(s_vec[f], t_vec[t])
            s = np.dot(c, l_vec[layer])
        except KeyError:
            s = random.sample([min_v, max_v], 1)[0]
        score.append(s)
        label.append(l)
    auc = roc_auc_score(label, score)
    print(auc)
    return auc


# 验证正反都用上的多任务
def verify_multi_G_GT(s_vec, t_vec, l_vec, T_s_vec, T_t_vec, T_l_vec, testPath):
    score = []
    label = []
    final_S = dict()
    final_T = dict()
    final_L = dict()
    for node in s_vec:
        final_S[node] = np.append(s_vec[node], T_t_vec[node])
        final_T[node] = np.append(t_vec[node], T_s_vec[node])

    for title in l_vec:
        final_L[title] = np.append(l_vec[title], T_l_vec[title])

    # 首先是获得测试集的网络名称
    for line in open(testPath):
        arr = line.strip().split(' ')
        if len(arr) == 3:
            try:
                layer = int(testPath.split('.test')[0].split('/')[-1])
            except ValueError:
                layer = int(testPath.split('.test')[
                            0].split('/')[-1].split('_')[-1])
        else:
            layer = int(line.strip().split(' ')[0])
        break
    max_v, min_v = get_max_min_for_multi(final_S, final_T, final_L[layer])
    for line in open(testPath):
        arr = line.strip().split(' ')
        if len(arr) == 3:
            try:
                layer = int(testPath.split('.test')[0].split('/')[-1])
            except ValueError:
                layer = int(testPath.split('.test')[
                            0].split('/')[-1].split('_')[-1])
            f = int(line.strip().split(' ')[0])
            t = int(line.strip().split(' ')[1])
            l = int(line.strip().split(' ')[2])
        else:
            layer = int(line.strip().split(' ')[0])
            f = int(line.strip().split(' ')[1])
            t = int(line.strip().split(' ')[2])
            l = int(line.strip().split(' ')[3])
        # if f not in s_vec or t not in t_vec or f not in T_t_vec or t not in T_s_vec:
        #     continue
        try:
            c = np.multiply(final_S[f], final_T[t])
            s = np.dot(c, final_L[layer])
        except KeyError:
            s = random.sample([min_v, max_v], 1)[0]
        score.append(s)
        label.append(l)
    auc = roc_auc_score(label, score)
    print(auc)
    return auc


def verify_strict_multi_G_GT(s_vec, t_vec, l_vec, T_s_vec, T_t_vec, T_l_vec, trainPath, testPath):
    """
    为了跟一般的方法采用公平的评价指标，对于单个网络上不可见的节点，随机赋值
    :param trainPath: 比如：7.train
    :param testPath: 比如：7.test
    :return:
    """
    score = []
    label = []
    final_S = dict()
    final_T = dict()
    final_L = dict()
    for node in s_vec:
        final_S[node] = np.append(s_vec[node], T_t_vec[node])
        final_T[node] = np.append(t_vec[node], T_s_vec[node])

    for title in l_vec:
        final_L[title] = np.append(l_vec[title], T_l_vec[title])

    # 首先是获得测试集的网络名称
    try:
        layer = int(testPath.split('.test')[0].split('/')[-1])
    except ValueError:
        layer = int(testPath.split('.test')[0].split('/')[-1].split('_')[-1])

    nodes = set()
    for line in open(trainPath):
        arr = line.strip().split(' ')
        nodes.add(int(arr[0]))
        nodes.add(int(arr[1]))
    max_v, min_v = get_max_min_for_multi(final_S, final_T, final_L[layer])
    for line in open(testPath):
        try:
            layer = int(testPath.split('.test')[0].split('/')[-1])
        except ValueError:
            layer = int(testPath.split('.test')[
                        0].split('/')[-1].split('_')[-1])
        f = int(line.strip().split(' ')[0])
        t = int(line.strip().split(' ')[1])
        l = int(line.strip().split(' ')[2])

        # if f not in s_vec or t not in t_vec or f not in T_t_vec or t not in T_s_vec:
        #     continue
        if f not in nodes or t not in nodes:
            s = random.sample([min_v, max_v], 1)[0]
        else:
            c = np.multiply(final_S[f], final_T[t])
            s = np.dot(c, final_L[layer])
        score.append(s)
        label.append(l)
    auc = roc_auc_score(label, score)
    print(auc)
    return auc


# 从一个文件中获取embedding
def get_embedding_from_path(f_path, dim, word2id):
    emb = Embedding(f_path, dim, word2id)
    model = dict()
    for word in word2id:
        model[word] = emb[str(word)]
    return model


# 从训练文件中获取词表
def get_word2id_from_file(f_path):
    nodes = set()
    titles = set()
    for line in open(f_path):
        arr = line.strip().split(' ')
        if len(arr) == 4:
            nodes.add(int(arr[1]))
            nodes.add(int(arr[2]))
            titles.add(int(arr[0]))
        else:
            nodes.add(int(arr[0]))
            nodes.add(int(arr[1]))
    node2id = dict(zip(sorted(map(int, nodes)), range(len(nodes))))
    title2id = dict(zip(sorted(map(int, titles)), range(len(titles))))
    if len(titles) == 0:
        title2id = None
    return node2id, title2id


def check_for_dne(base, title, m_name='asymmetric', dim=64):
    """
    验证DNE-model的效果
    :param base: 实验数据集的目录
    :param title: 某个网络
    :param m_name: 模型名称
    :param dim: embedding size
    """
    train_path = '../data/' + base + title + '.train'
    test_path = '../data/' + base + title + '.test'
    s_path = 'emb/' + base + m_name + '_' + title + '.emb_s'
    t_path = 'emb/' + base + m_name + '_' + title + '.emb_t'
    node2id, title2id = get_word2id_from_file(train_path)
    s_model = get_embedding_from_path(s_path, dim, node2id)
    t_model = get_embedding_from_path(t_path, dim, node2id)
    print('G上的效果为：')
    verify(s_model, t_model, '../data/' + base + title + '.test')

    T_s_path = 'emb/' + base + m_name + '_' + title + '_T.emb_s'
    T_t_path = 'emb/' + base + m_name + '_' + title + '_T.emb_t'
    T_s_model = get_embedding_from_path(T_s_path, dim, node2id)
    T_t_model = get_embedding_from_path(T_t_path, dim, node2id)
    print('最终的效果为：')
    verify_G_GT(s_model, t_model, T_s_model, T_t_model, test_path)


def try_check(base, title, m_name='asymmetric', dim=64):
    train_path = '../data/' + base + title + '.train'
    test_path = '../data/' + base + title + '.test'
    s_path = 'emb/' + base + m_name + '_' + title + '.emb_s'
    t_path = 'emb/' + base + m_name + '_' + title + '.emb_t'
    node2id, title2id = get_word2id_from_file(train_path)
    s_model = get_embedding_from_path(s_path, dim, node2id)
    t_model = get_embedding_from_path(t_path, dim, node2id)
    print('G上的效果为：')
    verify(s_model, t_model, '../data/' + base + title + '.test')

    T_s_path = 'emb/' + base + m_name + '_' + title + '_T.emb_s'
    T_t_path = 'emb/' + base + m_name + '_' + title + '_T.emb_t'
    T_s_model = get_embedding_from_path(T_s_path, dim, node2id)
    T_t_model = get_embedding_from_path(T_t_path, dim, node2id)
    for i in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
        print(np.dot(s_model[1], t_model[i]))


def check_for_multi_dne(base, input_name, dim=64, m_name='multi'):
    """
    验证multi_DNE的效果
    :param base: 实验数据集的目录
    :param input_name: 训练集的名称
    :param dim:
    :param m_name: 模型名称
    """
    train_path = '../data/' + base + input_name
    node2id, title2id = get_word2id_from_file(train_path)

    s_path = 'emb/' + base + m_name + '_' + input_name.split('.')[0] + '.emb_s'
    t_path = 'emb/' + base + m_name + '_' + input_name.split('.')[0] + '.emb_t'
    l_path = 'emb/' + base + m_name + '_' + input_name.split('.')[0] + '.emb_l'

    s_model = get_embedding_from_path(s_path, dim, node2id)
    t_model = get_embedding_from_path(t_path, dim, node2id)
    l_model = get_embedding_from_path(l_path, dim, title2id)

    type_list = [str(i) for i in title2id.keys()]
    performance = dict()
    for l in type_list:
        s = verify_multi(s_model, t_model, l_model,
                         '../data/' + base + l + '.test')
        performance[l] = s
    print('G上的效果为：')
    for k in performance:
        print(k, '\t', performance[k])

    T_s_path = 'emb/' + base + m_name + '_' + \
        input_name.split('.')[0] + '_T.emb_s'
    T_t_path = 'emb/' + base + m_name + '_' + \
        input_name.split('.')[0] + '_T.emb_t'
    T_l_path = 'emb/' + base + m_name + '_' + \
        input_name.split('.')[0] + '_T.emb_l'

    T_s_model = get_embedding_from_path(T_s_path, dim, node2id)
    T_t_model = get_embedding_from_path(T_t_path, dim, node2id)
    T_l_model = get_embedding_from_path(T_l_path, dim, title2id)

    type_list = [str(i) for i in title2id.keys()]
    performance = dict()
    for l in type_list:
        # s = verify_multi_G_GT(s_model, t_model, l_model, T_s_model, T_t_model, T_l_model,
        #                       '../data/' + base + l + '.test')
        s = verify_strict_multi_G_GT(s_model, t_model, l_model, T_s_model, T_t_model, T_l_model,
                                     '../data/' + base + l + '.train', '../data/' + base + l + '.test')
        performance[l] = s
    print('最终的效果为：')
    for k in performance:
        print(k, '\t', performance[k])


def check_for_parameter_mdne(base='LP_matrix_2015_40/', input_name='all.train', dim=64):
    """
    为参数敏感性实验的MDNE验证效果
    :param base:
    :param input_name:
    :return:
    """
    train_path = '../data/' + base + input_name
    node2id, title2id = get_word2id_from_file(train_path)

    s_path = 'Parameter_Sensitive_emb_MDNE/' + base + \
        'multi_all_dim_128_nsample_3_jrate0.85.emb_s'
    t_path = 'Parameter_Sensitive_emb_MDNE/' + base + \
        'multi_all_dim_128_nsample_3_jrate0.85.emb_t'
    l_path = 'Parameter_Sensitive_emb_MDNE/' + base + \
        'multi_all_dim_128_nsample_3_jrate0.85.emb_l'

    T_s_path = 'Parameter_Sensitive_emb_MDNE/' + base + \
        'multi_all_T_dim_128_nsample_3_jrate0.85.emb_s'
    T_t_path = 'Parameter_Sensitive_emb_MDNE/' + base + \
        'multi_all_T_dim_128_nsample_3_jrate0.85.emb_t'
    T_l_path = 'Parameter_Sensitive_emb_MDNE/' + base + \
        'multi_all_T_dim_128_nsample_3_jrate0.85.emb_l'

    s_model = get_embedding_from_path(s_path, dim, node2id)
    t_model = get_embedding_from_path(t_path, dim, node2id)
    l_model = get_embedding_from_path(l_path, dim, title2id)

    T_s_model = get_embedding_from_path(T_s_path, dim, node2id)
    T_t_model = get_embedding_from_path(T_t_path, dim, node2id)
    T_l_model = get_embedding_from_path(T_l_path, dim, title2id)

    type_list = [str(i) for i in title2id.keys()]
    performance = dict()
    for l in type_list:
        # s = verify_multi_G_GT(s_model, t_model, l_model, T_s_model, T_t_model, T_l_model,
        #                       '../data/' + base + l + '.test')
        s = verify_strict_multi_G_GT(s_model, t_model, l_model, T_s_model, T_t_model, T_l_model,
                                     '../data/' + base + l + '.train', '../data/' + base + l + '.test')
        performance[l] = s
    print('最终的效果为：')
    for k in performance:
        print(k, '\t', performance[k])


def check_for_verse(base, title, dim=128, m_name='verse'):
    """
    验证Verse效果
    :param base:
    :param input_name:
    :param dim:
    :param m_name:
    :return:
    """
    train_path = '../data/' + base + title + '.train'
    test_path = '../data/' + base + title + '.test'
    node2id, title2id = get_word2id_from_file(train_path)

    s_path = 'emb/' + base + m_name + '_' + title + '.emb'
    model = get_embedding_from_path(s_path, dim, node2id)
    auc = verify(model, model, test_path)
    print(auc)
    return auc


if __name__ == '__main__':
    # 针对VERSE的验证
    # title = '7'
    # base = 'LP_matrix_2015_50/'
    # m_name = 'verse'
    # check_for_verse(base, title)

    # 针对DNE的验证
    # title = '7'
    # base = 'LP_matrix_2015_50/'
    # m_name = 'asymmetric'
    # check_for_dne(base, title)

    title = '7'
    base = 'LP_matrix_2015_50/'
    m_name = 'asymmetric'
    try_check(base, title)

    # 针对多网络表征的验证
    # base = 'LP_matrix_2015_70/'
    # input_name = 'all.train'
    # dim = 64
    # m_name = 'multi'
    # check_for_multi_dne(base, input_name, dim=dim)

    # s_path = '../case_study/embeddings/asymmetric_7_all.emb_s'
    # node2id, _ = get_word2id_from_file('../data/All_matrix_2014/7_all')
    # get_embedding_from_path(s_path, 64, node2id)

    # 针对参数敏感性实验中的MDNE表征
    # base = 'LP_matrix_2015_80/'
    # check_for_parameter_mdne(base)

    # 针对app的表征
    # titles= ['5', '7', '18', '25']
    # for title in titles:
    #     node2id, _ = get_word2id_from_file('../data/LP_matrix_2015_80/'+title+'.train')
    #     s_path = 'emb/LP_matrix_2015_80/asymmetric_'+title+'.emb_s'
    #     t_path = 'emb/LP_matrix_2015_80/asymmetric_'+title+'.emb_t'
    #     s_model = get_embedding_from_path(s_path, 64, node2id)
    #     t_model = get_embedding_from_path(t_path, 64, node2id)
    #     test_path = '../data/LP_matrix_2015_80/'+title+'.test'
    #     auc = verify(s_model, t_model, test_path)
    #     print (title, auc)
