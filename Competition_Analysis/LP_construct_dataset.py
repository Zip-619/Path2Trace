import random
import os
import random


# from Baselines.Basic_Models import *
# from Baselines.Line import *
# from Baselines.Deepwalk import *
# from Baselines.Node2vec import *
# from Baselines.APP import *
# # from Baselines.Hope import *
# import Baselines.tools.Node2Vec_LayerSelect as Node2Vec_LayerSelect

# import Baselines.tools.Random_walk as Random_walk


def parse_args():
    parser = argparse.ArgumentParser(description="Run node2vec.")

    parser.add_argument('--input', nargs='?', default='data/matrix_2015',
                        help='Input graph path')
    # parser.add_argument('--output', nargs='?',default='../pengpai/city_graph/data/',
    #     help='output embedding path')

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
    返回：edge_data_by_type, all_edges, all_nodes, all_data
    """
    print('We are loading data from:', f_name)
    edge_data_by_type = dict()
    all_edges = list()
    all_nodes = list()
    all_data = dict()  # 用于存储所有边以及边对应的权重,每类网络的键值为 f_t
    all_data['Base'] = dict()
    with open(f_name, 'r') as f:
        for line in f:
            line = '1 '+line
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
            all_data[words[0]][key] = float(words[3])
            if key not in all_data['Base']:
                all_data['Base'][key] = 0
            all_data['Base'][key] += float(words[3])
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


if __name__ == '__main__':
    args = parse_args()
    file_name = args.input
    # outputpath = args.output
    args.proportion = 70

    save_dataset = True  # 是否保留数据集
    
    edge_data_by_type, _, all_nodes, all_edges_weight = load_network_data(file_name)
    proportion = args.proportion  # 正负样本比例
    training_data_by_type, evaluation_data_by_type = split_dataset(edge_data_by_type, proportion)

    for edge_type in training_data_by_type:
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
            simple_name = file_name # ../data/city_inmigration/data/edgelist/move_in/2020-01-10-edgelist.txt
            # data/edgelist/move_in/2021-01-30-edgelist.txt
            # if 'data/' in simple_name:
            #     simple_name = simple_name.split('data/')[1]
            # simple_name = simple_name.split('-edge')[0] # ../data/city_inmigration/data/edgelist/move_in/2020-01-10
            # base_foloder = outputpath
            t = simple_name.split('/')[-1].replace('-edgelist.txt','')
            move_type = simple_name.split('/')[-2]
            base_foloder = simple_name.split('/edgelist')[0]+'/LP_edgelist_'+str(proportion)
            base_foloder = os.path.join(base_foloder,move_type,t)
            # base_foloder = simple_name + '_' + str(proportion)
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
