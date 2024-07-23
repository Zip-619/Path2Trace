import argparse
import os
import random
import time

import Baselines.tools.Random_walk as Random_walk
from Baselines.APP import *
from Baselines.Basic_Models import *
from Baselines.Deepwalk import *
from Baselines.Line import *
from Baselines.Node2vec import *


# from Baselines.Hope import *


def parse_args():
    parser = argparse.ArgumentParser(description="Run node2vec.")

    parser.add_argument('--input', nargs='?', default='data/matrix_2015',
                        help='Input graph path')

    parser.add_argument('--output', nargs='?', default='emb/karate.emb',
                        help='Embeddings path')

    parser.add_argument('--dimensions', type=int, default=128,
                        help='Number of dimensions. Default is 100.')

    parser.add_argument('--walk-length', type=int, default=10,
                        help='Length of walk per source. Default is 80.')

    parser.add_argument('--num-walks', type=int, default=20,
                        help='Number of walks per source. Default is 10.')

    parser.add_argument('--window-size', type=int, default=10,
                        help='Context size for optimization. Default is 10.')

    parser.add_argument('--iter', type=int, default=10,
                        help='Number of epochs in SGD')

    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers. Default is 8.')

    parser.add_argument('--p', type=float, default=1,
                        help='Return hyperparameter. Default is 1.')

    parser.add_argument('--q', type=float, default=1,
                        help='Inout hyperparameter. Default is 1.')

    parser.add_argument('--weighted', dest='weighted', action='store_true',
                        help='Boolean specifying (un)weighted. Default is unweighted.')
    parser.add_argument('--unweighted', dest='unweighted', action='store_false')
    parser.set_defaults(weighted=True)

    parser.add_argument('--directed', dest='directed', action='store_true',
                        help='Graph is (un)directed. Default is undirected.')
    parser.add_argument('--undirected', dest='undirected', action='store_false')
    parser.add_argument('--proportion', type=int, default=20,
                        help='the proportion of training edges')
    parser.set_defaults(directed=True)

    return parser.parse_args()


def split_data(edge_data_by_type, train_proportion):
    """
    切割数据，按照一定的比例划分，原始图中每个节点至少被保留一条边

    """
    random.seed(2019)
    train_data_by_type = dict()
    evaluate_data_by_type = dict()
    for edge_type in edge_data_by_type:
        dataset = edge_data_by_type[edge_type]
        graph = dict()
        for edge in dataset:
            if edge[0] not in graph:
                graph[edge[0]] = []
            graph[edge[0]].append(edge[1])
        test_length = int(round(len(dataset) * (1 - train_proportion)))
        train_list = []
        evaluation_list = []
        random.shuffle(dataset)
        count = 0
        for data in dataset:
            if count == test_length:
                train_list.append(data)
            else:
                if len(graph[data[0]]) == 1:
                    train_list.append(data)
                else:
                    evaluation_list.append(data)
                    graph[data[0]].remove(data[1])
                    count += 1
        train_data_by_type[edge_type] = train_list
        evaluate_data_by_type[edge_type] = evaluation_list
        if len(evaluate_data_by_type[edge_type]) != test_length:
            print('期待的长度：', test_length, ' 真实拿到的长度：', len(evaluate_data_by_type[edge_type]), ' 相差：',
                  test_length - len(evaluate_data_by_type[edge_type]))
    return train_data_by_type, evaluate_data_by_type


def divide_edges(input_list, group_number=10):
    """
    给数据切割成10份
    """
    random.seed(2019)
    local_division = len(input_list) / float(group_number)
    random.shuffle(input_list)
    return [input_list[int(round(local_division * i)): int(round(local_division * (i + 1)))] for i in
            range(group_number)]


def split_dataset(edge_data_by_type, train_proportion):
    """
    划分数据集，一部分做训练集，剩下的做测试集
    """
    group_number = 10  # 将原始的数据划分成10份
    edge_data_by_type_separated = dict()
    for edge_type in edge_data_by_type:
        data = edge_data_by_type[edge_type]
        separated_data = divide_edges(data, 10)
        edge_data_by_type_separated[edge_type] = separated_data

    train_data_by_type = dict()
    evaluate_data_by_type = dict()
    split_number = int(train_proportion / group_number)

    for edge_type in edge_data_by_type_separated:
        train_data_by_type[edge_type] = list()
        evaluate_data_by_type[edge_type] = list()
        for i in range(0, group_number):
            if i < split_number:
                for tmp_edge in edge_data_by_type_separated[edge_type][i]:
                    train_data_by_type[edge_type].append((tmp_edge[0], tmp_edge[1]))
            else:
                for tmp_edge in edge_data_by_type_separated[edge_type][i]:
                    evaluate_data_by_type[edge_type].append((tmp_edge[0], tmp_edge[1]))
        print(edge_type, '训练样本的数目：', len(train_data_by_type[edge_type]), '. 测试样本的数目：',
              len(evaluate_data_by_type[edge_type]), '. 两者的比例为：',
              len(train_data_by_type[edge_type]) / len(evaluate_data_by_type[edge_type]))

    return train_data_by_type, evaluate_data_by_type


def load_network_data(f_name):
    """
    数据加载入口, 返回 edge_data_by_type 一个字典，key是类型，value是列表[(from_id, to_id)], 其中一个key是'base',其值是所有的边
    注意忽略了所有边之间的权重，并且创建了一个"Base"存储所有的边
    """
    print('We are loading data from:', f_name)
    edge_data_by_type = dict()
    all_edges = list()
    all_nodes = list()
    all_data = dict()  # 用于存储所有边以及边对应的权重,每类网络的键值为 f_t
    all_data['Base'] = dict()
    with open(f_name, 'r') as f:
        for line in f:
            words = line[:-1].split(' ')
            if words[0] not in edge_data_by_type:
                edge_data_by_type[words[0]] = list()
            edge_data_by_type[words[0]].append((words[1], words[2]))
            all_edges.append((words[1], words[2]))
            all_nodes.append(words[1])
            all_nodes.append(words[2])
            if words[0] not in all_data:
                all_data[words[0]] = {}
            key = words[1] + '_' + words[2]
            all_data[words[0]][key] = int(words[3])
            if key not in all_data['Base']:
                all_data['Base'][key] = 0
            all_data['Base'][key] += int(words[3])
    all_nodes = list(set(all_nodes))
    all_edges = list(set(all_edges))
    edge_data_by_type['Base'] = all_edges
    print('Finish loading data')
    return edge_data_by_type, all_edges, all_nodes, all_data


def get_G_from_edges(edges, edge_type, all_edges_weight):
    edge_dict = dict()
    for edge in edges:
        edge_key = str(edge[0]) + '_' + str(edge[1])
        edge_dict[edge_key] = all_edges_weight[edge_type][edge_key]
    tmp_G = nx.DiGraph()
    for edge_key in edge_dict:
        weight = edge_dict[edge_key]
        tmp_G.add_edge(edge_key.split('_')[0], edge_key.split('_')[1])
        tmp_G[edge_key.split('_')[0]][edge_key.split('_')[1]]['weight'] = weight
    return tmp_G


def randomly_choose_false_equal_edges(nodes, true_edges, number):
    """
    # 在训练集上产生产生与正样本等量的负样本
    # 下面两步是为了保证每次实验生成的负样本一样
    :param nodes: 训练集上的节点
    :param true_edges: 某种类型网络上所有的正边
    :param number: 已知正样例的个数
    """
    random.seed(2019)
    nodes = sorted(nodes)
    tmp_list = list()
    true_edges = set(true_edges)
    while number > 0:
        i = random.randint(0, len(nodes) - 1)
        j = random.randint(0, len(nodes) - 1)
        if i != j and (nodes[i], nodes[j]) not in true_edges:
            tmp_list.append((nodes[i], nodes[j]))
            number -= 1
    return tmp_list


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


def calculate_AUC(model, true_edges, false_edges, is_word2vec=True):
    """
    用于计算AUC，is_word2vec这直接从模型中取对应的结果，否则直接从字典中取
    之前计算AUC时，如果边的节点在训练集中没出现，就不计算AUC
    """
    true_list = list()
    prediction_list = list()
    if is_word2vec:
        nodes = set(model.wv.index2word)
        res = np.matmul(model.wv.syn0, model.wv.syn0.T)
        max_v = np.max(res)
        min_v = np.min(res)
    else:
        nodes = model.keys()
        max_v, min_v = get_max_min(model, model)
    for edge in true_edges:
        if edge[0] not in nodes or edge[1] not in nodes:
            # continue
            tmp_score = random.sample([min_v, max_v], 1)[0]
        else:
            tmp_score = calculate_score(model, edge[0], edge[1], is_word2vec)
        true_list.append(1)
        prediction_list.append(tmp_score)
    for edge in false_edges:
        if edge[0] not in nodes or edge[1] not in nodes:
            # continue
            tmp_score = random.sample([min_v, max_v], 1)[0]
        else:
            tmp_score = calculate_score(model, edge[0], edge[1], is_word2vec)
        true_list.append(0)
        prediction_list.append(tmp_score)
    y_true = np.array(true_list)
    y_scores = np.array(prediction_list)
    auc = roc_auc_score(y_true, y_scores)
    return auc


def calculate_score(model, node1, node2, is_word2vec=True):
    """
    从word2vec中直接取,或者从字典中取
    """
    if is_word2vec:
        vec1 = model.wv.syn0[model.wv.index2word.index(node1)]
        vec2 = model.wv.syn0[model.wv.index2word.index(node2)]
    else:
        vec1 = model[node1]
        vec2 = model[node2]
    s = np.dot(vec1, vec2)
    return s


if __name__ == '__main__':
    args = parse_args()
    file_name = args.input

    save_dataset = False  # 是否保留数据集

    edge_data_by_type, _, all_nodes, all_edges_weight = load_network_data(file_name)
    proportion = args.proportion  # 正负样本比例
    # training_data_by_type, evaluation_data_by_type = split_data(edge_data_by_type, proportion / 100.0)
    training_data_by_type, evaluation_data_by_type = split_dataset(edge_data_by_type, proportion)

    tmp_commonNeigh_performance = 0
    tmp_Jaccard_performance = 0
    tmp_AA_performance = 0
    tmp_preferential_performance = 0
    tmp_Deepwalk_performance = 0
    tmp_node2Vec_performance = 0
    tmp_LINE_performance = 0
    tmp_APP_performance = 0

    commonNeighbor_dic = dict()
    Jaccard_dic = dict()
    AA_dic = dict()
    preferential_dic = dict()
    deepwalk_dic = dict()
    node2vec_dic = dict()
    line_dic = dict()
    app_dic = dict()
    pmne1_dic = dict()
    pmne2_dic = dict()
    pmne3_dic = dict()

    for edge_type in ['7', '5', '18', '25']:
    # for edge_type in training_data_by_type:
        if edge_type == 'Base':
            continue

        print('We are working on edge:', edge_type)
        selected_true_edges = list()
        tmp_training_nodes = set()
        for edge in training_data_by_type[edge_type]:
            tmp_training_nodes.add(edge[0])
            tmp_training_nodes.add(edge[1])
        for edge in evaluation_data_by_type[edge_type]:
            tmp_training_nodes.add(edge[0])
            tmp_training_nodes.add(edge[1])
        for edge in evaluation_data_by_type[edge_type]:
            if edge[0] in tmp_training_nodes and edge[1] in tmp_training_nodes:
                if edge[0] == edge[1]:
                    continue
                selected_true_edges.append(edge)  # 保留验证集中的有效的边
        if len(selected_true_edges) == 0:
            continue
        print(edge_type, '正样例的个数为', len(selected_true_edges))

        selected_false_edges = randomly_choose_false_equal_edges(tmp_training_nodes, edge_data_by_type[edge_type], len(
            selected_true_edges))  # 产生的负例样本，训练节点的全链接所构成的所有样本，除去所有的正样本即可

        print('当前网络类型：', edge_type, '. 边数目为：', len(edge_data_by_type[edge_type]))
        print('训练样本总数：', len(training_data_by_type[edge_type]), '. 测试正样本为：', len(selected_true_edges), '. 测试负样本为：',
              len(selected_false_edges))

        ##########################################################################
        # 保存实验训练集和测试集
        if save_dataset:
            simple_name = file_name
            if 'data/' in simple_name:
                simple_name = simple_name.split('data/')[1]

            base_foloder = 'data/LP_' + simple_name + '_' + str(proportion)
            if not os.path.exists(base_foloder):
                os.makedirs(base_foloder)

            f = open(base_foloder + '/' + edge_type + '.train', 'w')
            f2 = open(base_foloder + '/' + edge_type + '_T.train', 'w')
            for edge in training_data_by_type[edge_type]:
                f.write(
                    edge[0] + ' ' + edge[1] + ' ' + str(all_edges_weight[edge_type][edge[0] + '_' + edge[1]]) + '\n')
                f2.write(
                    edge[1] + ' ' + edge[0] + ' ' + str(all_edges_weight[edge_type][edge[0] + '_' + edge[1]]) + '\n')
            f.close()
            f2.close()
            f = open(base_foloder + '/' + edge_type + '.test', 'w')
            for edge in selected_true_edges:
                f.write(edge[0] + ' ' + edge[1] + ' 1\n')
            for edge in selected_false_edges:
                f.write(edge[0] + ' ' + edge[1] + ' 0\n')
            f.close()
            continue

        #########################################################################
        # 以下是common_neighbor的实现
        tmp_commonNeigh_score = common_neighbor(training_data_by_type[edge_type], selected_true_edges,
                                                selected_false_edges)
        tmp_commonNeigh_performance += tmp_commonNeigh_score
        commonNeighbor_dic[edge_type] = tmp_commonNeigh_score
        print("Common Neighbor的结果为:", tmp_commonNeigh_score)
        ############################################################################

        ############################################################################
        # 以下是Jaccard的实现
        tmp_Jaccard_score = Jaccard(training_data_by_type[edge_type], selected_true_edges,
                                    selected_false_edges)
        tmp_Jaccard_performance += tmp_Jaccard_score
        Jaccard_dic[edge_type] = tmp_Jaccard_score
        print("Jaccard的结果为:", tmp_Jaccard_score)
        ############################################################################

        ############################################################################
        # 以下是AA的实现
        tmp_AA_score = Adamic_Adar(training_data_by_type[edge_type], selected_true_edges,
                                   selected_false_edges)
        tmp_AA_performance += tmp_AA_score
        AA_dic[edge_type] = tmp_AA_score
        print("AA的结果为:", tmp_AA_score)
        ############################################################################

        ############################################################################
        # 以下是preferential的实现
        tmp_preferential_score = preferential(training_data_by_type[edge_type], selected_true_edges,
                                              selected_false_edges)
        tmp_preferential_performance += tmp_preferential_score
        preferential_dic[edge_type] = tmp_preferential_score
        print("Preferential的结果为:", tmp_preferential_score)
        ##########################################################################

        ############################################################################
        # 以下是Line的实现
        LINE_model = train_LINE_model(training_data_by_type[edge_type], all_edges_weight[edge_type])
        tmp_LINE_score = calculate_AUC(LINE_model, selected_true_edges, selected_false_edges, False)
        tmp_LINE_performance += tmp_LINE_score
        line_dic[edge_type] = tmp_LINE_score
        print("Line的结果为:", tmp_LINE_score)
        # ############################################################################

        ###########################################################################
        # 以下是deepwalk的实现
        Deepwalk_G = Random_walk.RWGraph(
            get_G_from_edges(training_data_by_type[edge_type], edge_type, all_edges_weight), args.directed, 1, 1)

        t1 = time.time()
        Deepwalk_G.preprocess_transition_probs()
        t2 = time.time()
        print('DeepWalk 构建转移概率时间 = ', (t2 - t1))
        Deepwalk_walks = Deepwalk_G.simulate_walks(args.num_walks, 10)
        t3 = time.time()
        print('Deepwalk随机游走时间 = ', (t3 - t2))
        Deepwalk_model = train_deepwalk_embedding(Deepwalk_walks)
        tmp_Deepwalk_score = calculate_AUC(Deepwalk_model, selected_true_edges, selected_false_edges)
        tmp_Deepwalk_performance += tmp_Deepwalk_score
        deepwalk_dic[edge_type] = tmp_Deepwalk_score
        print('Deepwalk的结果为:' + str(tmp_Deepwalk_score))
        ###########################################################################

        ###########################################################################
        # 以下是node2vec的实现，主要思想就是用node2vec的规则产生sentence，然后执行word2vec
        node2vec_G = Random_walk.RWGraph(
            get_G_from_edges(training_data_by_type[edge_type], edge_type, all_edges_weight), args.directed, 2, 0.5)
        t1 = time.time()
        node2vec_G.preprocess_transition_probs()
        t2 = time.time()
        print('node2vec构建转移概率时间 = ', (t2 - t1))
        node2vec_walks = node2vec_G.simulate_walks(20, 10)
        t3 = time.time()
        print('node2vec随机游走时间 = ', (t3 - t2))
        node2vec_model = train_node2vec_embedding(node2vec_walks)
        tmp_node2vec_score = calculate_AUC(node2vec_model, selected_true_edges, selected_false_edges)
        tmp_node2Vec_performance += tmp_node2vec_score
        node2vec_dic[edge_type] = tmp_node2vec_score
        print('Node2vec的结果为:' + str(tmp_node2vec_score))
        ###########################################################################

        ##########################################################################
        # 以下是APP的实现
        # APP_model = train_App_model(train_edges=training_data_by_type[edge_type], all_edges=all_edges_weight[edge_type])
        # tmp_APP_score = get_AUC_App(APP_model, selected_true_edges, selected_false_edges)
        # tmp_APP_performance += tmp_APP_score
        # app_dic[edge_type] = tmp_APP_score
        # print(edge_type, '\tAPP的结果为:' + str(tmp_APP_score))
        ##########################################################################

    ############################################################################
    # 保留大图
    if save_dataset:
        f = open(base_foloder + '/all.train', 'w')
        f2 = open(base_foloder + '/all_T.train', 'w')
        for edge_type in training_data_by_type:
            if edge_type == 'Base':
                continue
            for line in open(base_foloder + '/' + edge_type + '.train'):
                f.write(edge_type + ' ' + line)
            for line in open(base_foloder + '/' + edge_type + '_T.train'):
                f2.write(edge_type + ' ' + line)
        f.close()
        f2.close()
    ############################################################################

    all_types = list()
    # for k in training_data_by_type:
    for k in ['5', '18', '7', '25']:
        if k == 'Base':
            continue
        all_types.append(k)
    print('proportion = ', proportion)
    print('Model\t' + '\t'.join(all_types))
    print('CommonNeigh\t' + '\t'.join([str(commonNeighbor_dic[title]) for title in all_types]))
    print('Jaccard\t' + '\t'.join([str(AA_dic[title]) for title in all_types]))
    print('AA\t' + '\t'.join([str(commonNeighbor_dic[title]) for title in all_types]))
    print('Preferential\t' + '\t'.join([str(preferential_dic[title]) for title in all_types]))
    print('Deepwalk\t' + '\t'.join([str(deepwalk_dic[title]) for title in all_types]))
    print('Node2vec\t' + '\t'.join([str(node2vec_dic[title]) for title in all_types]))
    print('Line\t' + '\t'.join([str(line_dic[title]) for title in all_types]))
    # print('APP\t' + '\t'.join([str(app_dic[title]) for title in all_types]))

    f = open('result', 'a')
    f.write(time.strftime("%Y-%m-%d %H:%M:%S",
                          time.localtime()) + '\nInput File = ' + file_name + '. Train proportion: ' + str(
        proportion) + '\n')
    f.write('Model\t' + '\t'.join(all_types) + '\n')
    f.write('CommonNeigh\t' + '\t'.join([str(commonNeighbor_dic[title]) for title in all_types]) + '\n')
    f.write('Jaccard\t' + '\t'.join([str(AA_dic[title]) for title in all_types]) + '\n')
    f.write('AA\t' + '\t'.join([str(commonNeighbor_dic[title]) for title in all_types]) + '\n')
    f.write('Preferential\t' + '\t'.join([str(preferential_dic[title]) for title in all_types]) + '\n')
    f.write('Deepwalk\t' + '\t'.join([str(deepwalk_dic[title]) for title in all_types]) + '\n')
    f.write('Node2vec\t' + '\t'.join([str(node2vec_dic[title]) for title in all_types]) + '\n')
    f.write('Line\t' + '\t'.join([str(line_dic[title]) for title in all_types]) + '\n')
    # f.write('APP\t' + '\t'.join([str(app_dic[title]) for title in all_types])+'\n')
    f.close()
