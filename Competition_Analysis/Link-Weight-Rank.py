import argparse
import pickle

from DNE.embedding import *


def parse_args():
    parser = argparse.ArgumentParser(description="Run Verse.")

    parser.add_argument('--model', nargs='?', default='asymmetric',
                        help='which model to use')

    parser.add_argument('--base', nargs='?', default='Link_rank_15_17',
                        help='path of folder')

    parser.add_argument('--title', nargs='?', default='7',
                        help='which network to use')

    parser.add_argument('--k', type=int, default=5,
                        help='the parameter of ndcg')

    parser.add_argument('--is_map', type=str, default='False')

    return parser.parse_args()


def load_data(f_name):
    """
    加载数据，这边的数据入口是一个网络
    :param f_name: 数据类型 f t w
    :return: 
    """
    edges = dict()  # 存储所有节点的出边，f : [t1, t2, t3 ...]
    out_degree = dict()  # 存储所有节点的出度之和 f : out_degree
    weights = dict()  # 存储所有节点的出度之和 f : out_degree
    for line in open(f_name):
        words = line.strip().split(' ')
        if words[0] not in edges:
            edges[words[0]] = list()
        edges[words[0]].append(words[1])
        if words[0] not in out_degree:
            out_degree[words[0]] = 0
        out_degree[words[0]] += int(words[-1])
        key = words[0] + '_' + words[1]
        if key not in weights:
            weights[key] = 0
        weights[key] += int(words[-1])
    return edges, out_degree, weights


def calculate_map(truth, prediction):
    """
    计算MAP的值
    :param truth: ground truth
    :param prediction:
    """
    index_list = list()
    for element in truth:
        if element not in prediction:
            continue
        e_index = prediction.index(element) + 1
        index_list.append(e_index)
    index_list = sorted(index_list)
    ap = 0
    for i in range(0, len(index_list)):
        ap += (i + 1) / index_list[i]
    return ap / len(truth)


def concat_embedding(s_vec, t_vec, T_s_vec, T_t_vec):
    """
    拼接向量
    """
    final_s_vec = dict()
    final_t_vec = dict()
    for node in s_vec:
        final_s_vec[node] = np.append(s_vec[node], T_t_vec[node])
        final_t_vec[node] = np.append(t_vec[node], T_s_vec[node])

    return final_s_vec, final_t_vec


def evaluate_map(s_vec, t_vec, test_path='data/Link_rank_15_17/7.try'):
    """
    验证map效果
    """
    test_dic = {}
    for line in open(test_path):
        arr = line.strip().split(' ')
        f_node = int(arr[0])
        t_node = int(arr[1])
        if f_node not in s_vec or t_node not in t_vec:  # 如果节点没在训练集中出现，直接跳过
            continue
        if f_node not in test_dic:
            test_dic[f_node] = list()
        test_dic[f_node].append(t_node)
    print('实际测试样例的数目 = ', len(test_dic))
    ap_list = list()
    predict_dic = dict()

    count = 0
    for f_node in test_dic:
        count += 1
        predict_dic[f_node] = dict()
        for t_node in t_vec:
            score = np.dot(s_vec[f_node], t_vec[t_node])
            predict_dic[f_node][t_node] = score
        ls = sorted(predict_dic[f_node].items(), key=lambda x: x[1], reverse=True)
        pre_list = list()
        for l in ls:
            pre_list.append(l[0])
        ap = calculate_map(test_dic[f_node], pre_list)
        # print(f_node, ap)
        ap_list.append(ap)
        # if count % 100 == 0:
        #     print('第%d轮, 当前MAP均值等于%f'%(count, np.mean(np.asarray(ap_list))))
    return np.mean(np.array(ap_list))


def evaluate_map_multi(s_vec, t_vec, l_vec, title=7, test_path='data/Link_rank_15_17/7.try'):
    """
    验证DNE-multi map效果
    """
    test_dic = {}
    for line in open(test_path):
        arr = line.strip().split(' ')
        f_node = int(arr[0])
        t_node = int(arr[1])
        if f_node not in s_vec or t_node not in t_vec:  # 如果节点没在训练集中出现，直接跳过
            continue
        if f_node not in test_dic:
            test_dic[f_node] = list()
        test_dic[f_node].append(t_node)
    ap_list = list()
    predict_dic = dict()

    count = 0
    print(l_vec.keys())
    for f_node in test_dic:
        count += 1
        predict_dic[f_node] = dict()
        for t_node in t_vec:
            tmp_score = np.multiply(s_vec[f_node], t_vec[t_node])
            score = np.dot(tmp_score, l_vec[title])
            predict_dic[f_node][t_node] = score
        ls = sorted(predict_dic[f_node].items(), key=lambda x: x[1], reverse=True)
        pre_list = list()
        for l in ls:
            pre_list.append(l[0])
        ap = calculate_map(test_dic[f_node], pre_list)
        # print(f_node, ap)
        ap_list.append(ap)
        # if count % 100 == 0:
        #     print('第%d轮, 当前MAP均值等于%f'%(count, np.mean(np.asarray(ap_list))))
    return np.mean(np.array(ap_list))


def basic_evaluate_map(train_path='data/Link_Rank_14_16_17/7_2014_2016',
                       test_path='data/Link_Rank_14_16_17/7_2017_2020'):
    """
    直接用训练集的数据去预测测试集
    :param train_path: 原图的数据
    :param test_path: 测试的位置
    :return: 返回map的结果
    """

    test_dic = {}
    for line in open(test_path):
        arr = line.strip().split(' ')
        f_node = int(arr[0])
        t_node = int(arr[1])
        if f_node not in test_dic:
            test_dic[f_node] = list()
        test_dic[f_node].append(t_node)

    print('实际测试样例的数目 = ', len(test_dic))
    ap_list = list()
    predict_dic = {}

    for line in open(train_path):
        arr = line.strip().split(' ')
        f_node = int(arr[0])
        t_node = int(arr[1])
        if f_node not in predict_dic:
            predict_dic[f_node] = dict()
        if t_node not in predict_dic[f_node]:
            predict_dic[f_node][t_node] = 0
        predict_dic[f_node][t_node] = int(arr[-1])

    count = 0
    for f_node in test_dic:
        count += 1
        if f_node not in predict_dic:
            continue
        tmp_list = sorted(predict_dic[f_node].items(), key=lambda x: x[1], reverse=True)
        predict_list = list()
        for tmp in tmp_list:
            predict_list.append(tmp[0])
        ap = calculate_map(test_dic[f_node], predict_list)
        # print(f_node, ap)
        ap_list.append(ap)
        # if count % 100 == 0:
        #     print('第%d轮, 当前map均值等于%f' % (count, np.mean(np.asarray(ap_list))))

    return np.mean(np.array(ap_list))


def dcg_at_k(r, k, method=0):
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0


def ndcg_at_k(r, k, method=0):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0
    return dcg_at_k(r, k, method) / dcg_max


def get_reverse_list(f_node, s_vec, t_vec):
    """
    给定一个节点，根据embedding求出该点与其他节点之间的score,并进行排序。这边主要针对的是用两个向量点积衡量score的方法
    :param f_node: 给你一个节点
    :return: 返回节点list，按照与目标节点之间的score倒序排列
    """
    if f_node not in s_vec or f_node not in t_vec:
        print('node is illegal')
        return []
    pred_dic = dict()
    for t_node in t_vec:
        if f_node == t_node:
            continue
        pred_dic[t_node] = np.dot(s_vec[f_node], t_vec[t_node])
    pred_list = sorted(pred_dic.items(), key=lambda x: x[1], reverse=True)
    res_list = list()
    for l in pred_list:
        res_list.append(l[0])
    return res_list


def get_reverse_list_for_multi(f_node, title, s_vec, t_vec, l_vec):
    """
    为multi表征做的排序，不同之处在计算score时用三个向量
    """
    if f_node not in s_vec or f_node not in t_vec:
        print('node is illegal')
        return []
    pred_dic = dict()
    for t_node in t_vec:
        if f_node == t_node:
            continue
        pred_dic[t_node] = np.dot(np.multiply(s_vec[f_node], t_vec[t_node]), l_vec[title])
    pred_list = sorted(pred_dic.items(), key=lambda x: x[1], reverse=True)
    res_list = list()
    for l in pred_list:
        res_list.append(l[0])
    return res_list


def read_vectors(file_name):
    tmp_embedding = dict()
    file = open(file_name, 'r')
    for line in file.readlines()[1:]:
        if len(line.strip().split(' ')) == 2:
            continue
        numbers = line[:-2].split(' ')
        tmp_vector = list()
        for n in numbers[1:]:
            tmp_vector.append(float(n))
            tmp_embedding[numbers[0]] = np.asarray(tmp_vector)
    file.close()
    return tmp_embedding


def basic_evaluate_ndcg(test_dict, train_path='data/Link_rank_15_17/18_all', k_number=5):
    train_dic = dict()
    nodes = set()
    for line in open(train_path):
        arr = line.strip().split(' ')
        f_node = int(arr[0])
        t_node = int(arr[1])
        weight = int(arr[2])
        nodes.add(f_node)
        nodes.add(t_node)
    for f_node in nodes:
        if f_node not in train_dic:
            train_dic[f_node] = dict()
        for t_node in nodes:
            train_dic[f_node][t_node] = 0
    for line in open(train_path):
        arr = line.strip().split(' ')
        f_node = int(arr[0])
        t_node = int(arr[1])
        weight = int(arr[2])
        train_dic[f_node][t_node] = weight

    count = 0
    size = 0
    final_ndcg = list()
    nodes = list(test_dict.keys())
    random.shuffle(nodes)
    # for f_node in test_dict:
    for f_node in nodes:
        size += 1
        if f_node not in train_dic:
            continue
        pred_node = sorted(train_dic[f_node].items(), key=lambda x: x[1], reverse=True)
        tmp_list = list()
        for l in pred_node:
            tmp_list.append(l[0])
        pred_node = tmp_list
        pred_list = list()
        for node in pred_node:
            if node in test_dict[f_node]:
                pred_list.append(test_dict[f_node][node])
            else:
                pred_list.append(0)
        ndcg = ndcg_at_k(pred_list, k_number)
        if ndcg > 0:
            count += 1
        final_ndcg.append(ndcg)
        if size % 1000 == 0:
            print('第%d轮, 当前ndcg@%d均值等于%f' % (size, k_number, np.mean(np.asarray(final_ndcg))))
    print('验证集总个数：', len(final_ndcg), ', 其中大于0的个数', count)
    return np.mean(np.asarray(final_ndcg))


def evaluate_ndcg(test_dict, s_emb, t_emb, k_number=5):
    """
    分别为每个节点先算，最后再去取平均
    """
    count = 0
    size = 0
    final_ndcg = list()
    nodes = list(test_dict.keys())
    random.shuffle(nodes)
    # for f_node in test_dict:
    for f_node in nodes:
        size += 1
        if f_node not in s_emb:
            continue
        pred_node = get_reverse_list(f_node, s_emb, t_emb)  # 按照score倒排之后的结果
        pred_list = list()
        for node in pred_node:
            if node in test_dict[f_node]:
                pred_list.append(test_dict[f_node][node])
            else:
                pred_list.append(0)
        ndcg = ndcg_at_k(pred_list, k_number)
        if ndcg > 0:
            count += 1
        # print(f_node, ndcg)
        final_ndcg.append(ndcg)
        if size % 1000 == 0:
            print('第%d轮, 当前ndcg@%d均值等于%f' % (size, k_number, np.mean(np.asarray(final_ndcg))))
    ndcg = np.mean(np.asarray(final_ndcg))
    print(ndcg)
    print('验证集总个数：', len(final_ndcg), ', 其中大于0的个数', count)
    return ndcg


def evaluate_ndcg_for_multi(test_dict, s_emb, t_emb, l_emb, title=7, k_number=5):
    """
    为多任务模型学习ndcg
    """
    count = 0
    size = 0
    final_ndcg = list()
    nodes = list(test_dict.keys())
    random.shuffle(nodes)
    # for f_node in test_dict:
    for f_node in nodes:
        size += 1
        if f_node not in s_emb:
            continue
        pred_node = get_reverse_list_for_multi(f_node, title, s_emb, t_emb, l_emb)  # 按照score倒排之后的结果
        pred_list = list()
        for node in pred_node:
            if node in test_dict[f_node]:
                pred_list.append(test_dict[f_node][node])
            else:
                pred_list.append(0)
        ndcg = ndcg_at_k(pred_list, k_number)
        if ndcg > 0:
            count += 1
        # print(f_node, ndcg)
        final_ndcg.append(ndcg)
        if size % 1000 == 0:
            print('第%d轮, 当前ndcg@%d均值等于%f' % (size, k_number, np.mean(np.asarray(final_ndcg))))
    ndcg = np.mean(np.asarray(final_ndcg))
    print(ndcg)
    print('验证集总个数：', len(final_ndcg), ', 其中大于0的个数', count)
    return ndcg


def get_dne_embedding():
    base_path = 'DNE/emb/' + args.base + '/asymmetric_' + args.title + '_all'
    s_path = base_path + '.emb_s'
    t_path = base_path + '.emb_t'
    T_s_path = base_path + '_T.emb_s'
    T_t_path = base_path + '_T.emb_t'
    train_path = 'data/' + args.base + '/' + args.title + '_all'
    node2id, _ = get_word2id_from_file(train_path)
    dim = 64
    s_vec = get_embedding_from_path(s_path, dim, node2id)
    t_vec = get_embedding_from_path(t_path, dim, node2id)
    T_s_vec = get_embedding_from_path(T_s_path, dim, node2id)
    T_t_vec = get_embedding_from_path(T_t_path, dim, node2id)
    final_s_vec, final_t_vec = concat_embedding(s_vec, t_vec, T_s_vec, T_t_vec)
    s_emb = final_s_vec
    t_emb = final_t_vec
    return s_emb, t_emb


def get_mdne_emebdding():
    base_path = 'DNE/emb/' + args.base + '/multi_all_all'
    s_path = base_path + '.emb_s'
    t_path = base_path + '.emb_t'
    l_path = base_path + '.emb_l'
    T_s_path = base_path + '_T.emb_s'
    T_t_path = base_path + '_T.emb_t'
    T_l_path = base_path + '_T.emb_l'
    train_path = 'data/' + args.base + '/all_all'
    node2id, title2id = get_word2id_from_file(train_path)
    s_vec = get_embedding_from_path(s_path, 64, node2id)
    t_vec = get_embedding_from_path(t_path, 64, node2id)
    l_vec = get_embedding_from_path(l_path, 64, title2id)
    T_s_vec = get_embedding_from_path(T_s_path, 64, node2id)
    T_t_vec = get_embedding_from_path(T_t_path, 64, node2id)
    T_l_vec = get_embedding_from_path(T_l_path, 64, title2id)
    final_s_vec, final_t_vec = concat_embedding(s_vec, t_vec, T_s_vec, T_t_vec)
    final_l_vec = dict()
    for title in l_vec:
        final_l_vec[title] = np.append(l_vec[title], T_l_vec[title])
    return final_s_vec, final_t_vec, final_l_vec


def get_verse_embedding():
    train_path = 'data/' + args.base + '/' + args.title + '_all'
    emb_path = 'DNE/emb/' + args.base + '/verse_' + args.title + '_all.emb'
    node2id, _ = get_word2id_from_file(train_path)
    emb = get_embedding_from_path(emb_path, 128, node2id)
    return emb


def get_app_embedding():
    base_path = 'DNE/emb/' + args.base + '/asymmetric_' + args.title + '_all'
    s_path = base_path + '.emb_s'
    t_path = base_path + '.emb_t'
    T_s_path = base_path + '_T.emb_s'
    T_t_path = base_path + '_T.emb_t'
    train_path = 'data/' + args.base + '/' + args.title + '_all'
    node2id, _ = get_word2id_from_file(train_path)
    dim = 64
    s_vec = get_embedding_from_path(s_path, dim, node2id)
    t_vec = get_embedding_from_path(t_path, dim, node2id)
    s_emb = s_vec
    t_emb = t_vec
    return s_emb, t_emb


def get_deepwalk_embedding():
    with open('emb/deepwalk_LR_' + args.title + '_all.emb', 'rb') as f:
        deepwalk_emb = pickle.load(f)
    emb = dict()
    for node in deepwalk_emb:
        emb[int(node)] = deepwalk_emb[node]
    return emb


def get_node2vec_embedding():
    with open('emb/node2vec_LR_' + args.title + '_all.emb', 'rb') as f:
        node2vec_emb = pickle.load(f)
    emb = dict()
    for node in node2vec_emb:
        emb[int(node)] = node2vec_emb[node]
    return emb


def get_line_embedding():
    with open('emb/line_LR_' + title + '_all.emb', 'rb') as f:
        Line_emb = pickle.load(f)
    emb = dict()
    for node in Line_emb:
        emb[int(node)] = Line_emb[node]
    return emb


def get_hope_embedding():
    with open('emb/hope_LR_' + title + '_all.emb_s', 'rb') as f:
        s_model = pickle.load(f)
    with open('emb/hope_LR_' + title + '_all.emb_t', 'rb') as f:
        t_model = pickle.load(f)
    s_emb = dict()
    t_emb = dict()
    for node in s_model:
        s_emb[int(node)] = s_model[node]
        t_emb[int(node)] = t_model[node]
    return s_emb, t_emb


def get_gcn_ae_embedding():
    with open('emb/gcn_ae_LR_' + title + '.emb', 'rb') as f:
        gae_emb = pickle.load(f)
    emb = dict()
    for node in gae_emb:
        emb[int(node)] = gae_emb[node]
    return emb


def get_gcn_vae_embedding():
    with open('emb/gcn_vae_LR_' + title + '.emb', 'rb') as f:
        gae_emb = pickle.load(f)
    emb = dict()
    for node in gae_emb:
        emb[int(node)] = gae_emb[node]
    return emb


def get_pmne_embedding():
    with open('emb/pmne_LR.emb', 'rb') as f:
        pmne_emb = pickle.load(f)
    emb = dict()
    for node in pmne_emb:
        emb[int(node)] = pmne_emb[node]
    return emb


def get_mne_embedding(title):
    with open('emb/MNE_LR.emb', 'rb') as f:
        MNE_model = pickle.load(f)
    local_model = dict()
    for pos in range(len(MNE_model['index2word'])):
        local_model[MNE_model['index2word'][pos]] = MNE_model['base'][pos] + 0.1 * np.dot(
            MNE_model['addition'][title][pos], MNE_model['tran'][title])
    emb = dict()
    for node in local_model:
        emb[int(node)] = local_model[node]
    return emb


if __name__ == '__main__':

    args = parse_args()
    k_number = args.k
    title = args.title
    model = args.model
    if args.is_map == 'False':
        is_map = False
        print('now execute NDCG')
    if args.is_map == 'True':
        is_map = True
        print('now execute MAP')

    # test_path = 'data/' + args.base + '/' + title + '.test'

    test_path = 'data/' + args.base + '/' + title + '.try'
    test_dict = dict()  # 存储每个节点的出边以及权重
    for line in open(test_path):
        arr = line.strip().split(' ')
        f_node = int(arr[0])
        t_node = int(arr[1])
        weight = int(arr[2])
        if f_node not in test_dict:
            test_dict[f_node] = dict()
        if t_node not in test_dict[f_node]:
            test_dict[f_node][t_node] = weight

    if model == 'dne':
        s_emb, t_emb = get_dne_embedding()
        if not is_map:
            ndcg = evaluate_ndcg(test_dict, s_emb, t_emb, k_number=k_number)
            print('DNE NDCG@' + str(k_number) + ' = ', ndcg)
        if is_map:
            map = evaluate_map(s_emb, t_emb, test_path)
            print('DNE MAP = ', map)

    if model == 'mdne':
        final_s_vec, final_t_vec, final_l_vec = get_mdne_emebdding()
        if not is_map:
            ndcg = evaluate_ndcg_for_multi(test_dict, final_s_vec, final_t_vec, final_l_vec, title=int(args.title),
                                           k_number=k_number)
            print('MDNE NDCG@%d = %f' % (k_number, ndcg))
        if is_map:
            map = evaluate_map_multi(final_s_vec, final_t_vec, final_l_vec, title=int(args.title), test_path=test_path)
            print('MDNE MAP = ', map)

    if model == 'basic':
        train_path = 'data/' + args.base + '/' + args.title + '_all'
        if not is_map:
            ndcg = basic_evaluate_ndcg(test_dict, k_number=k_number, train_path=train_path)
            print('原图 NDCG@%d = %f' % (k_number, ndcg))
        if is_map:
            map = basic_evaluate_map(train_path=train_path, test_path=test_path)
            print('原图 MAP = ', map)

    if model == 'verse':
        emb = get_verse_embedding()
        if not is_map:
            ndcg = evaluate_ndcg(test_dict, emb, emb, k_number=k_number)
            print('Verse NDCG@%d = %f' % (k_number, ndcg))
        if is_map:
            map = evaluate_map(emb, emb)
            print('Verse MAP = ', map)

    if model == 'app':
        s_emb, t_emb = get_app_embedding()
        if not is_map:
            ndcg = evaluate_ndcg(test_dict, s_emb, t_emb, k_number=k_number)
            print('app NDCG@%d = %f' % (k_number, ndcg))
        if is_map:
            map = evaluate_map(s_emb, t_emb)
            print('app MAP = ', map)

    if model == 'deepwalk':
        emb = get_deepwalk_embedding()
        if not is_map:
            ndcg = evaluate_ndcg(test_dict, emb, emb, k_number=k_number)
            print('Deepwalk NDCG@%d = %f' % (k_number, ndcg))
        if is_map:
            map = evaluate_map(emb, emb)
            print('Deepwalk MAP = ', map)

    if model == 'node2vec':
        emb = get_node2vec_embedding()
        if not is_map:
            ndcg = evaluate_ndcg(test_dict, emb, emb, k_number=k_number)
            print('Node2vec NDCG@%d = %f' % (k_number, ndcg))
        if is_map:
            map = evaluate_map(emb, emb)
            print('Node2vec MAP = ', map)

    if model == 'line':
        emb = get_line_embedding()
        if not is_map:
            ndcg = evaluate_ndcg(test_dict, emb, emb, k_number=k_number)
            print('Line NDCG@%d = %f' % (k_number, ndcg))
        if is_map:
            map = evaluate_map(emb, emb)
            print('Line MAP = ', map)

    if model == 'hope':
        s_emb, t_emb = get_hope_embedding()
        if not is_map:
            ndcg = evaluate_ndcg(test_dict, s_emb, t_emb, k_number=k_number)
            print('Hope NDCG@%d = %f' % (k_number, ndcg))
        if is_map:
            map = evaluate_map(emb, emb)
            print('Hope MAP = ', map)

    if model == 'gae':
        emb = get_gcn_ae_embedding()
        if not is_map:
            ndcg = evaluate_ndcg(test_dict, emb, emb, k_number=k_number)
            print('GCN_AE NDCG@%d = %f' % (k_number, ndcg))
        if is_map:
            map = evaluate_map(emb, emb)
            print('GCN_AE MAP = ', map)

    if model == 'gvae':
        emb = get_gcn_vae_embedding()
        if not is_map:
            ndcg = evaluate_ndcg(test_dict, emb, emb, k_number=k_number)
            print('GCN_VAE NDCG@%d = %f' % (k_number, ndcg))
        if is_map:
            map = evaluate_map(emb, emb)
            print('GCN_VAE MAP = ', map)

    if model == 'pmne':
        emb = get_pmne_embedding()
        if not is_map:
            ndcg = evaluate_ndcg(test_dict, emb, emb, k_number=k_number)
        print('PMNE NDCG@%d = %f' % (k_number, ndcg))
        if is_map:
            map = evaluate_map(emb, emb)
            print('PMNE MAP = ', map)

    if model == 'mne':
        emb = get_mne_embedding(args.title)
        if not is_map:
            ndcg = evaluate_ndcg(test_dict, emb, emb, k_number=k_number)
        print('MNE NDCG@%d = %f' % (k_number, ndcg))
        if is_map:
            map = evaluate_map(emb, emb)
            print('MNE MAP = ', map)


    if not is_map:
        f = open('ndcg_link_weight_rank', 'a')
        f.write(args.model + ' ' + args.title + ' ' + 'ndcg@' + str(k_number) + ' ' + str(ndcg) + '\n')
        f.close()
    else:
        f = open('map_link_weight_rank', 'a')
        f.write(args.model + ' ' + args.title + ' map: ' + str(map) + '\n')
        f.close()
