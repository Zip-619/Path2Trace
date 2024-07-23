'''
Author: Zip
Date: 2021-11-22 23:21:03
LastEditors: Zip
LastEditTime: 2021-12-06 14:01:41
Description: 保存一些通用的函数
'''
import json
import logging
import math
import pickle
import random
from urllib.request import urlopen, quote

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
from bidict import bidict
from sklearn import metrics
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, \
                    datefmt='%Y/%m/%d %H:%M:%S', \
                    format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s')
logger = logging.getLogger(__name__)
HEAD = 0
TAIL = 1


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_neg_neb(neg_count, node, nebs, old2new_case_id, case_info, sample_type):
    case_city = case_info.loc[old2new_case_id.inv[node]]['confirm_city']
    case_province = case_city.strip('/')[0]
    if sample_type == 'same_city':
        case_pool = list(set(case_info.loc[case_info['confirm_city'] == case_city].index).difference(set(nebs)))
    elif sample_type == 'random':
        case_pool = list(set(case_info.index).difference(set(nebs)))
    elif sample_type == 'same_province':
        case_pool = list(
            set(case_info.loc[case_info['confirm_city'].str.contains(case_province)].index).difference(set(nebs)))
    new_case_id_pool = [
        old2new_case_id[old_case_id] for old_case_id in case_pool
        # old2new_case_id 有问题
    ]
    if len(new_case_id_pool) <= neg_count:
        return list(new_case_id_pool)
    else:
        return random.sample(new_case_id_pool, neg_count)


def generate_pos_neg_samples(config, net, old2new_case_id):
    """
    根据spread net 以及config中的sample type，生成正负样本对

    Args:
        config: parser 对象
        net: spread net， networkx 对象
        old2new_case_id: 病例id映射

    Returns:
        all_idx_pair: 病例idx对及其label，
    """
    sample_type = config.sample_type

    proportion = config.proportion
    pos_idx_pair = []
    neg_idx_pair = []
    case_info = pd.read_excel('../data/pengpai/labeled_data/1114-completed.xlsx', index_col='idx')
    part_nodes = random.sample(set(net.nodes()), int(proportion * len(net.nodes())))

    for node in part_nodes:
        nebs = list(net.neighbors(node))
        for neb in nebs:
            pos_idx_pair.append([node,neb,1])  # 正样本
        neg_pair_count = int(round(random.uniform(1,config.lbd)))  # 根据lambda参数，设置负样本的个数
        if neg_pair_count<=0:  # 对于每个节点，都添加若干个负样本节点，组成负样本对
            neg_idx = get_neg_neb(neg_count=1,nebs = nebs,old2new_case_id=old2new_case_id,case_info=case_info,sample_type='random')
        else:
            neg_idx = get_neg_neb(neg_pair_count,node,nebs,old2new_case_id,case_info,sample_type)
        for neg in neg_idx:
            neg_idx_pair.append([node,neg,0])

    all_idx_pair = pos_idx_pair + neg_idx_pair
    print('sample length = '+str(len(all_idx_pair)))
    with open('dataset/case_idx_pair/all_idx_pair{}_{}.pickle'.format(str(config.lbd), sample_type), 'wb') as file:
        pickle.dump(file=file, obj=all_idx_pair)
        file.close()

    return all_idx_pair


def get_emb_label(padded_tensor_list, index_batch_x, device):
    """通过index_batch_x中的index pair，在padded_tensor_list中获得对应的emb
    Args:
        padded_tensor_list ([list(tensor)]): [padded case path tensor]
        index_batch_x ([[[]]]): [index_batch_x: pos and neg case pair and labels]
        device ([torch.device]): [train device]

    Returns:
        [tuple(path1,path2,y)]: [case path embs with batch]
    """
    y = []
    tensor_pairs = []
    i = 0
    length = len(padded_tensor_list[index_batch_x[0][i]])
    while (i < len(index_batch_x[0])):  # batch len
        tensor_1 = padded_tensor_list[index_batch_x[0][i]]
        tensor_2 = padded_tensor_list[index_batch_x[1][i]]
        y.append(index_batch_x[2][i])
        tensor_pair = torch.cat((tensor_1, tensor_2))
        tensor_pairs.append(tensor_pair)
        i += 1

    path1 = torch.stack(tensor_pairs)[:, :length].to(device)
    path2 = torch.stack(tensor_pairs)[:, length:].to(device)

    y = torch.tensor(y, dtype=torch.long).to(device)
    return path1, path2, y

def save_pickle(obj, filepath):
    with open(filepath,'wb') as file:
        pickle.dump(obj,file,protocol=2)
        file.close()

def get_index(tensor, tenser_list):
    """get a index of a tensor from a tensor list

    Args:
        tensor (tensor): tensor need to find
        tenser_list (list(tensor)): tensor list

    Returns:
        int: tensor index
    """
    for index, data in enumerate(tenser_list):
        if torch.equal(data, tensor):
            return index
    assert index != len(tenser_list) - 1, 'not found tensor'


def get_case_tensor_dict(case_paths, poi_emb):
    """ get the tensor path of case

    Args:
        case_paths (dict): dict of case and it's address obj list
        poi_emb (dict): poi address to embedding learned by representation

    Returns:
        dict: a dict with key of case id and value of path addr tensors
    """
    with open('../data/pengpai/labeled_data/idx_address_network_no_edgedata/nodeidxmaps.pickle', 'rb') as file:
        address_node2idx = pickle.load(file)
    poi2idxbi = bidict(address_node2idx['poi'])
    # print('poi to index 的双向字典已经建立')

    case_path_emb = {}
    for case_id, path in case_paths.items():
        case_path_emb[case_id] = []
        try:
            for addr in path:
                if addr.get_addr('poi') == '':
                    continue
                else:
                    case_path_emb[case_id].append(torch.Tensor(poi_emb[addr.get_addr('poi')]))
        except Exception as e:
            print('出现异常的地点：', end='')
            print(e)
    # print('获得了所有的病例的出行路径及其对应的表示学习的embedding')
    case_tensor_dict = {}
    for case, path_emb in case_path_emb.items():
        try:
            case_tensor_dict[case] = torch.squeeze(torch.stack(path_emb).reshape((1, -1)))
        except Exception as e:  # 少数异常地点使用随机向量替代
            case_tensor_dict[case] = torch.squeeze(torch.rand(1,
                                                              128))  # 先临时随机生成一个，出错的病例已经在/data/GAOZip/Case_Association_Prediction/data/pengpai/labeled_data/1110-merge-修改.xlsx中改了过来，等有时间重新构建一下网络，学习表征
    # print('已经所有的病例的出行路径压缩成一个向量')
    return case_tensor_dict


amap_ak = ''# amap tooken
def amap_addr_query(address):
    """query a string of address with amap query api

    Args:
        address (str): address

    Returns:
        json: query result
    """
    # 该方法在高德地图中查询输入的地址，返回JSon格式的查询结果

    url = 'https://restapi.amap.com/v3/geocode/geo?'
    query = quote(address)
    output = 'json'
    url2 = url + '&address=' + query + '&output=' + output + '&key=' + amap_ak
    req = urlopen(url2, timeout=10)
    res = req.read().decode()
    query_result = json.loads(res)
    return query_result


def load_file(file_path):
    """load file with 'rb' type

    Args:
        file_path (str): file path

    Returns:
        file obj: file load result
    """
    with open(file_path, 'rb') as file:
        case_paths = pickle.load(file)
        file.close()
    return case_paths


def sort_reindex_case_tensor_dict(case_tensor_dict):
    """sort case tensor by length of tensor and reindex case tensor. Return the reindexed case tensor and the map between old case id and reindexed case id

    Args:
        case_tensor_dict (dict): dict of case id and case path tensor

    Returns:
        reindexed_case_tensor_dict: reindexed case tensor dict
        old2new_case_id : map between old and new case id
    """
    case_tensor_dict_sorted_unpadded = sorted(case_tensor_dict.items(), key=lambda x: len(x[1]), reverse=True)
    #  (11241, tensor([ 1.6260,  1.1816, -0.0308,  ...,  1.4752,  1.3250,  0.8201])),
    #  (4987, tensor([ 1.6648, -0.2088, -0.8032,  ..., -1.6231, -0.3634,  0.1699])),
    #  (7634, tensor([ 1.3742,  0.1653, -0.9707,  ..., -0.7406, -0.0693,  0.1163])),

    # 由于病例的caseID不是连续的（对病例有筛选），所以应当重新建立索引，用于构建dataset：
    i = 0
    old2new_case_id = bidict()  # 用于保存旧的和新的caseID之间的映射
    reindexed_case_tensor_unpadded = []
    for old_case_ID, tensor in case_tensor_dict_sorted_unpadded:
        reindexed_case_tensor_unpadded.append(tensor)
        old2new_case_id[old_case_ID] = i
        # if old_case_ID == 1894:
        #     print("1894 1894")
        i += 1
    return reindexed_case_tensor_unpadded, old2new_case_id


def reindex_sp_net(spread_net, old2new_case_id):
    """reindex sp_net by using old2new_case_id

    Args:
        spread_net (networkx): spread network
        old2new_case_id (dict): mapping between old and new case id 
    Returns:
        networkx : reindexed sp network 
    """
    reindexed_sp_net = nx.Graph()
    for node in spread_net.nodes():
        reindexed_sp_net.add_node(old2new_case_id[node])
    for edge in spread_net.edges():
        reindexed_sp_net.add_edge(old2new_case_id[edge[HEAD]], old2new_case_id[edge[TAIL]])
    return reindexed_sp_net


def get_all_loc(case_paths):
    case_paths = load_file('../data/pengpai/labeled_data/case_path.pickle')
    poi_addr_bidict = {}
    for case_id, path in case_paths.items():
        for addr in path:
            poi_addr = addr.get_addr('poi')
            poi_addr_bidict[addr] = poi_addr
    addr2loc = {}
    logger.info('Querying poi address logitude and latitude...')
    for _, val in tqdm(poi_addr_bidict.items()):
        if val in addr2loc.keys():
            continue
        query_res = amap_addr_query(val)
        if query_res['status'] == '1' and len(query_res['geocodes']) != 0:
            if query_res['geocodes'][0]['location'] != '':
                addr2loc[val] = {}
                loc_str = query_res['geocodes'][0]['location']
                addr2loc[val]['log'] = float(loc_str.split(',')[0])
                addr2loc[val]['lat'] = float(loc_str.split(',')[1])
    with open('../data/pengpai/labeled_data/poi_addr_loc.pickle', 'wb') as file:
        pickle.dump(file=file, obj=addr2loc)
        file.close()
    return addr2loc

def find_optimal_cutoff_roc(TPR,FPR,Threshold):
    """
    get the optimal cutoff threshold based on youden index
    Args:
        TPR:
        FPR:
        Threshold:

    Returns:

    """
    y = TPR-FPR
    Youden_index = np.argmax(y)
    optimal_threshold = Threshold[Youden_index]
    point = [FPR[Youden_index],TPR[Youden_index]]
    return optimal_threshold,point

def ROC(label, y_prob):
    """
    Receiver_Operating_Characteristic, ROC
    :param label: (n, )
    :param y_prob: (n, )
    :return: fpr, tpr, roc_auc, optimal_th, optimal_point
    """
    fpr, tpr, thresholds = metrics.roc_curve(label, y_prob)
    roc_auc = metrics.auc(fpr, tpr)
    optimal_th, optimal_point = find_optimal_cutoff_roc(TPR=tpr, FPR=fpr, Threshold=thresholds)
    return fpr, tpr, roc_auc, optimal_th, optimal_point

def draw_roc(y,y_hat):
    fpr, tpr, roc_auc, optimal_th, optimal_point = ROC(y, y_hat)
    plt.figure(1)
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.plot(optimal_point[0], optimal_point[1], marker='o', color='r')
    plt.text(optimal_point[0], optimal_point[1], f'FPR:{optimal_point[0]:.2f},TPR:{optimal_point[1]:.2f}')
    plt.title("ROC-AUC")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.show()
    return roc_auc, fpr, tpr

def aggregate_importance(importances):
    return sum(importances)/len(importances)

def get_case_tensor(place_emb_method):

    case_paths = load_file('../data/pengpai/labeled_data/case_path.pickle')
    print('load poi embedding from rl method: {}'.format(place_emb_method))

    poi_embedding = load_file('../represent_learning/poi_embeddings/'+place_emb_method+'_poi_embedding.pickle')

    print('poi embedding shape: {}'.format(len(list(poi_embedding.values())[1])))
    assert len(case_paths) == 8604, 'missing {} case'.format((8604 - len(case_paths)))
    case_tensor_dict = get_case_tensor_dict(case_paths, poi_embedding)
    reindexed_case_tensor_unpadded, old2new_case_id = sort_reindex_case_tensor_dict(case_tensor_dict)
    assert len(old2new_case_id.keys()) == 8604, 'missing {} case'.format(8604 - len(old2new_case_id))
    reindexed_case_tensor_padded = pad_sequence(reindexed_case_tensor_unpadded, batch_first=True)
    spread_net = load_file('spread_net.pickle')
    reindexed_sp_net = reindex_sp_net(spread_net, old2new_case_id)
    return reindexed_sp_net,old2new_case_id,reindexed_case_tensor_padded