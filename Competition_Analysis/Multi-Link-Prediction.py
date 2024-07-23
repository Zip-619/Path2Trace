import argparse
import random
import time

import Baselines.tools.Random_walk as Random_walk
from Baselines.APP import *
from Baselines.Basic_Models import *
from Baselines.Deepwalk import *
from Baselines.Line import *
from Baselines.MNE import *
from Baselines.Node2vec import *


def parse_args():
    parser = argparse.ArgumentParser(description="Run node2vec.")

    parser.add_argument('--input', nargs='?', default='data/matrix_2015',
                        help='Input graph path')

    parser.add_argument('--output', nargs='?', default='emb/karate.emb',
                        help='Embeddings path')

    parser.add_argument('--dimensions', type=int, default=200,
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
    parser.add_argument('--proportion', type=int, default=50,
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


def calculate_AUC(model, true_edges, false_edges, is_word2vec=True):
    """
    用于计算AUC，is_word2vec这直接从模型中取对应的结果，否则直接从字典中取
    之前计算AUC时，如果边的节点在训练集中没出现，就不计算AUC
    """
    true_list = list()
    prediction_list = list()
    if is_word2vec:
        nodes = set(model.wv.index2word)
    else:
        nodes = model.keys()
    count = 0
    for edge in true_edges:
        if edge[0] not in nodes or edge[1] not in nodes:
            count += 1
            continue
        tmp_score = calculate_score(model, edge[0], edge[1], is_word2vec)
        true_list.append(1)
        prediction_list.append(tmp_score)
    for edge in false_edges:
        if edge[0] not in nodes or edge[1] not in nodes:
            continue
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


def Evaluate_PMNE_methods(input_network):
    # we need to write codes to implement the co-analysis method of PMNE
    print('Start to analyze the PMNE method')
    training_network = input_network['training']
    test_network = input_network['test_true']
    false_network = input_network['test_false']
    all_network = list()
    all_test_network = list()
    all_false_network = list()
    all_nodes = set([])
    for edge_type in training_network:
        for edge in training_network[edge_type]:
            all_network.append(edge)
            if edge[0] not in all_nodes:
                all_nodes.add(edge[0])
            if edge[1] not in all_nodes:
                all_nodes.add(edge[1])
        for edge in test_network[edge_type]:
            all_test_network.append(edge)
        for edge in false_network[edge_type]:
            all_false_network.append(edge)
    all_nodes = list(all_nodes)
    #############################################################################################
    # PME-one
    model_one_performance = dict()
    # all_network = set(all_network)
    # G = Random_walk.RWGraph(get_G_from_edges(all_network, 'Base', all_edges_weight), args.directed, args.p,
    #                         args.q)  # 这个地方使用的是全部的边
    # G.preprocess_transition_probs()
    # walks = G.simulate_walks(args.num_walks, args.walk_length)
    # model_one = train_deepwalk_embedding(walks)
    # # method_one_performance = get_AUC(model_one, all_test_network,
    # #                                  all_false_network)  # 我觉得这个地方有问题，这边现在就相当于把所有的训练集合并，将所有的正负样本合并
    # # print('Performance of PMNE method one:', method_one_performance)
    # for edge_type in test_network:
    #     score = calculate_AUC(model_one, test_network[edge_type], false_network[edge_type])
    #     model_one_performance[edge_type] = score
    # for key in model_one_performance:
    #     print('PMNE-1', key, '\t', model_one_performance[key])
    #############################################################################################
    # PMNE-two
    model_two_performance = dict()
    all_models = list()
    for edge_type in training_network:
        tmp_edges = training_network[edge_type]
        tmp_G = Random_walk.RWGraph(get_G_from_edges(tmp_edges, edge_type, all_edges_weight), args.directed, args.p,
                                    args.q)
        tmp_G.preprocess_transition_probs()
        walks = tmp_G.simulate_walks(args.num_walks, args.walk_length)
        tmp_model = train_deepwalk_embedding(walks)
        all_models.append(tmp_model)
    model_two = merge_PMNE_models(all_models, all_nodes)
    # method_two_performance = get_dict_AUC(model_two, all_test_network, all_false_network)
    # print('Performance of PMNE method two:', method_two_performance)
    for edge_type in test_network:
        score = calculate_AUC(model_two, test_network[edge_type], false_network[edge_type], is_word2vec=False)
        model_two_performance[edge_type] = score
    for key in model_two_performance:
        print('PMNE-2', key, '\t', model_two_performance[key])
    #############################################################################################
    # PMNE-three
    model_three_performance = dict()
    # tmp_graphs = list()
    # for edge_type in training_network:
    #     tmp_G = get_G_from_edges(training_network[edge_type], edge_type, all_edges_weight)
    #     tmp_graphs.append(tmp_G)
    # MK_G = Node2Vec_LayerSelect.Graph(tmp_graphs, args.p, args.q, 0.5)  # 这里实现的就是PMNE(c)里面的随机游走的策略
    # MK_G.preprocess_transition_probs()
    # MK_walks = MK_G.simulate_walks(args.num_walks, args.walk_length)
    # model_three = train_deepwalk_embedding(MK_walks)
    # # method_three_performance = get_AUC(model_three, all_test_network, all_false_network)
    # # print('Performance of PMNE method three:', method_three_performance)
    # for edge_type in test_network:
    #     score = calculate_AUC(model_three, test_network[edge_type], false_network[edge_type])
    #     model_three_performance[edge_type] = score
    # for key in model_three_performance:
    #     print('PMNE-3', key, '\t', model_three_performance[key])
    return model_one_performance, model_two_performance, model_three_performance


def merge_PMNE_models(input_all_models, all_nodes):
    final_model = dict()
    for tmp_model in input_all_models:
        for node in all_nodes:
            if node in final_model:
                if node in tmp_model.wv.index2word:
                    final_model[node] = np.concatenate(
                        (final_model[node], tmp_model.wv.syn0[tmp_model.wv.index2word.index(node)]), axis=0)
                else:
                    final_model[node] = np.concatenate((final_model[node], np.zeros([args.dimensions])), axis=0)
            else:
                if node in tmp_model.wv.index2word:
                    final_model[node] = tmp_model.wv.syn0[tmp_model.wv.index2word.index(node)]
                else:
                    final_model[node] = np.zeros([args.dimensions])
    return final_model


def get_dict_AUC(model, true_edges, false_edges):
    true_list = list()
    prediction_list = list()
    for edge in true_edges:
        tmp_score = get_dict_neighbourhood_score(model, str(edge[0]), str(edge[1]))
        true_list.append(1)
        # prediction_list.append(tmp_score)
        # for the unseen pair, we randomly give a prediction
        if tmp_score > 2:
            if tmp_score > 2.5:
                prediction_list.append(1)
            else:
                prediction_list.append(-1)
        else:
            prediction_list.append(tmp_score)
    for edge in false_edges:
        tmp_score = get_dict_neighbourhood_score(model, str(edge[0]), str(edge[1]))
        true_list.append(0)
        # prediction_list.append(tmp_score)
        # for the unseen pair, we randomly give a prediction
        if tmp_score > 2:
            if tmp_score > 2.5:
                prediction_list.append(1)
            else:
                prediction_list.append(-1)
        else:
            prediction_list.append(tmp_score)
    y_true = np.array(true_list)
    y_scores = np.array(prediction_list)
    return roc_auc_score(y_true, y_scores)


def get_dict_neighbourhood_score(local_model, node1, node2):
    try:
        vector1 = local_model[node1]
        vector2 = local_model[node2]
        return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    except:
        return 2 + random.random()


if __name__ == '__main__':
    args = parse_args()
    file_name = args.input

    save_dataset = False # 是否保留数据集

    edge_data_by_type, _, all_nodes, all_edges_weight = load_network_data(file_name)
    proportion = args.proportion  # 正负样本比例
    # training_data_by_type, evaluation_data_by_type = split_data(edge_data_by_type, proportion / 100.0)
    training_data_by_type, evaluation_data_by_type = split_dataset(edge_data_by_type, proportion)


    pmne1_dic = dict()
    pmne2_dic = dict()
    pmne3_dic = dict()

    mne_dic = dict()
    mne_dic2 = dict()

    # MNE_model = train_model(training_data_by_type, all_edges_weight)

    base_edges = list()
    training_nodes = list()  # 这里的training_nodes指的是出现在所有网络上的节点
    for edge_type in training_data_by_type:
        for edge in training_data_by_type[edge_type]:
            base_edges.append(edge)
            training_nodes.append(edge[0])
            training_nodes.append(edge[1])
    training_nodes = list(set(training_nodes))
    training_data_by_type['Base'] = base_edges
    #
    merged_networks = dict()
    merged_networks['training'] = dict()
    merged_networks['test_true'] = dict()
    merged_networks['test_false'] = dict()

    for edge_type in training_data_by_type:
        if edge_type == 'Base':
            continue

        print('We are working on edge:', edge_type)
        selected_true_edges = list()
        tmp_training_nodes = list()
        for edge in training_data_by_type[edge_type]:
            tmp_training_nodes.append(edge[0])
            tmp_training_nodes.append(edge[1])
        tmp_training_nodes = set(tmp_training_nodes)
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

        ############################################################################
        # 以下是MNE的实现部分
        # local_model = dict()
        # for pos in range(len(MNE_model['index2word'])):
        #     local_model[MNE_model['index2word'][pos]] = MNE_model['base'][pos] + 0.5 * np.dot(
        #         MNE_model['addition'][edge_type][pos], MNE_model['tran'][edge_type])
        # tmp_MNE_score = get_dict_AUC(local_model, selected_true_edges, selected_false_edges)
        # print(edge_type, ' MNE performance:', tmp_MNE_score)
        # mne_dic[edge_type] = tmp_MNE_score
        #
        # tmp_MNE_score_2 = calculate_AUC(local_model, selected_true_edges, selected_false_edges, False)
        # print(edge_type, ' MNE performamce2', tmp_MNE_score_2)
        # mne_dic2[edge_type] = tmp_MNE_score_2
        ############################################################################

        merged_networks['training'][edge_type] = set(training_data_by_type[edge_type])
        merged_networks['test_true'][edge_type] = selected_true_edges
        merged_networks['test_false'][edge_type] = selected_false_edges

    pmne1_dic, pmne2_dic, pmne3_dic = Evaluate_PMNE_methods(merged_networks)

    all_types = list()
    for k in training_data_by_type:
        if k == 'Base':
            continue
        all_types.append(k)


    f = open('result', 'a')
    f.write('################################\n')
    f.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + '\nInput File = ' + file_name + '. Train proportion: ' + str(proportion) + '\n')
    f.write('Model\t' + '\t'.join(all_types)+'\n')

    # f.write('MNE1\t' + '\t'.join([str(mne_dic[title]) for title in all_types]) + '\n')
    # f.write('MNE2\t' + '\t'.join([str(mne_dic2[title]) for title in all_types]) + '\n')
    #
    # f.write('PMNE1\t' + '\t'.join([str(pmne1_dic[title]) for title in all_types])+'\n')
    f.write('PMNE2\t' + '\t'.join([str(pmne2_dic[title]) for title in all_types])+'\n')
    # f.write('PMNE3\t' + '\t'.join([str(pmne3_dic[title]) for title in all_types])+'\n')
    print('proportion = ', proportion)
    print('Model\t' + '\t'.join(all_types))
    # print('MNE1\t' + '\t'.join([str(mne_dic[title]) for title in all_types]) + '\n')
    # print('MNE2\t' + '\t'.join([str(mne_dic2[title]) for title in all_types]) + '\n')
    # print('PMNE1\t' + '\t'.join([str(pmne1_dic[title]) for title in all_types]))
    print('PMNE2\t' + '\t'.join([str(pmne2_dic[title]) for title in all_types]))
    # print('PMNE3\t' + '\t'.join([str(pmne3_dic[title]) for title in all_types]))
    f.close()

