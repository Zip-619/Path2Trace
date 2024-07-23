"""
Calculate the importance of the places based on attention mechanism, mask the location based on the importance of the location, and compare it with randomly mask
"""

import argparse
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from dataloader.dataloader import get_dataloader
from utils import *
import torch.utils.data
from model.minmaxmodel import *

import random
from utils import setup_seed, aggregate_importance


def sep_place(path_list):
    return path_list[:len(path_list) // 2], path_list[len(path_list) // 2:]



def train(train_dataloader, config, reindexed_case_tensor_padded):
    """
    train process
    :param train_dataloader:
    :param config:
    :param reindexed_case_tensor_padded:
    :return: trained model
    """
    device = torch.device('cuda:{}'.format(config.gpu) if torch.cuda.is_available() else 'cpu')
    LP_model = LP(128, config.path_len)
    opt_LP = torch.optim.Adam(LP_model.parameters(), lr=config.lr_LP)

    LP_loss = nn.MSELoss()
    LP_model.to(device)

    epochs = tqdm(range(config.num_epochs), ncols=100)
    count = 0
    for _ in epochs:
        for batch_x in train_dataloader:
            path1, path2, y = get_emb_label(reindexed_case_tensor_padded, batch_x, device)

            opt_LP.zero_grad()
            path1 = path1.reshape([128, 45, 128]).permute(1, 0, 2)
            path2 = path2.reshape([128, 45, 128]).permute(1, 0, 2)
            # path1 shape torch.Size([45, 128, 128])
            # attn_output: (L, N, E)(L,N,E) where L is the target sequence length(45),
            # N is the batch size(128), E is the embedding dimension.
            # (N, L, E)(N,L,E) if batch_first is True.
            #
            # attn_output_weights: (N, L, S)(N,L,S) where N is the batch size,
            # L is the target sequence length, S is the source sequence length.
            y_hat, path1_emb, path2_emb, _, _ = LP_model(path1, path2)
            y = y.float()
            y_hat = y_hat.squeeze()
            LP_l = LP_loss(y, y_hat)
            LP_l.backward()
            opt_LP.step()
            count += 1

        epochs.set_postfix(LP_loss=LP_l.cpu().item())

    # torch.save(LP_model.state_dict(),
    #            '{}/{}_{}_{}_model.pth'.format(config.model_save_dir, config.model_name, 'mask', config.epo))

    return LP_model


def test(test_dataloader, config, LP_model, reindexed_case_tensor_padded):
    """
    test process
    :param test_dataloader:
    :param config:
    :param LP_model:
    :param reindexed_case_tensor_padded:
    :return:
    """
    y, y_hat = [], []
    device = torch.device("cuda:{}".format(config.gpu) if torch.cuda.is_available() else 'cpu')
    LP_model.to(device)
    for batch_x in test_dataloader:
        path1, path2, y_label = get_emb_label(reindexed_case_tensor_padded, batch_x, device)
        with torch.no_grad():
            path1 = path1.reshape([128, 45, 128]).permute(1, 0, 2)
            path2 = path2.reshape([128, 45, 128]).permute(1, 0, 2)
            y_hat_out, _, _, _, _ = LP_model(path1, path2)
            y_hat_out = y_hat_out.squeeze()
            y_label = y_label.float()
        y_hat += y_hat_out.cpu()
        y += y_label.cpu()

    auc, fpr, tpr = draw_roc(y, y_hat)

    return auc


def fileter_place(dataloader, config, LP_model, reindexed_case_tensor_padded):
    """
    Obtain all locations' importance
    :param dataloader:
    :param config:
    :param LP_model:
    :param reindexed_case_tensor_padded:
    :return:
    """
    device = torch.device('cuda:{}'.format(config.gpu)) if torch.cuda.is_available() else 'cpu'
    LP_model.to(device)
    LP_model.eval()
    count = 0
    for batch_x in dataloader:
        path1, path2, y_label = get_emb_label(reindexed_case_tensor_padded, batch_x, device)
        with torch.no_grad():
            path1 = path1.reshape([128, 45, 128]).permute(1, 0, 2)
            path2 = path2.reshape([128, 45, 128]).permute(1, 0, 2)
            y_hat_out, path1_emb, path2_emb, path1_weight, path2_weight = LP_model(path1, path2)
            path1_weight = path1_weight.reshape([-1, 45, 45])
            path2_weight = path2_weight.reshape([-1, 45, 45])
            path1_mean = (path1_weight * torch.log(path1_weight)).sum(dim=2)
            path2_mean = (path2_weight * torch.log(path2_weight)).sum(dim=2)

            path_place_importance = torch.cat((path1_mean.reshape(-1), path2_mean.reshape(-1)))
            path = torch.cat((path1.reshape(128, 45, -1), path2.reshape(128, 45, -1)), 0)

            if count != 0:
                total_place_importance = torch.cat((total_place_importance, path_place_importance), 0)
                total_place_emb = torch.cat((total_place_emb, path), 0)
            else:
                total_place_emb = path
                total_place_importance = path_place_importance

            count += 1
    total_place_emb = total_place_emb.reshape([-1, 128]).cpu()
    total_place_importance = total_place_importance.reshape([-1, 1]).cpu()
    assert total_place_importance.shape[0] == total_place_emb.shape[0]
    total_place_emb, idx = total_place_emb.unique(dim=0, return_inverse=True)
    total_place_emb_importance = torch.cat((total_place_emb, torch.empty((total_place_emb.shape[0], 1))), 1)

    temp_dic = {}

    for i, item in enumerate(idx.tolist()):
        if item not in temp_dic.keys():
            temp_dic[item] = [total_place_importance[i].item()]
        else:
            temp_dic[item] += [total_place_importance[i].item()]

    for key, val in temp_dic.items():
        place_importance = aggregate_importance(val)
        total_place_emb_importance[key][-1] = place_importance


    assert total_place_emb_importance.shape[0] == len(temp_dic.keys())


    return total_place_emb, total_place_importance, total_place_emb_importance


def filter_tensor(total_place_emb_importance, path1, device, topk, count, mask_ratio):
    """
    filter place embeddings according to the place importance.
    :param total_place_emb_importance: dict of embedding to importance
    :param path1: Paths to be filtered
    :param device:
    :param topk:
    :param count:
    :param mask_ratio:
    :return: filtered path tensor
    """
    temp_importance = torch.Tensor().to(device)
    for place in path1:
        try:
            temp_importance = torch.cat([temp_importance, total_place_emb_importance[
                torch.nonzero(torch.eq(total_place_emb_importance[:, :-1], place).all(dim=1)).item()][-1].unsqueeze(
                dim=0)])
        except ValueError:
            temp_importance = torch.cat([temp_importance, torch.Tensor([0]).to(device)])
            count += 1
            # print(place)
    a = 0
    path1_emb_importance = torch.cat([path1, temp_importance.unsqueeze(dim=1)], dim=1)
    path1_emb_importance = path1_emb_importance.reshape([-1, 45, 129])

    path1 = path1.reshape([-1, 45, 128])
    val_place = torch.nonzero(path1.sum(dim=2)).tolist()
    val_len_dic = {}
    for i in range(len(val_place)):
        if val_place[i][0] not in val_len_dic.keys():
            val_len_dic[val_place[i][0]] = 1
        else:
            val_len_dic[val_place[i][0]] += 1
    for i in range(len(path1)):
        mask_c = val_len_dic[i]
        indice = torch.topk(path1_emb_importance[i, :mask_c, -1], max(1, round(mask_c * mask_ratio)), largest=False)[
            1].tolist()
        for j in indice:
            path1[i, j, :] = torch.zeros(128).to(device)

    return path1


def filter_tensor_random(total_place_emb_importance, path1, device, topk, count, mask_ratio):
    """
    filter place embeddings randomly
    :param total_place_emb_importance:
    :param path1:
    :param device:
    :param topk:
    :param count:
    :param mask_ratio:
    :return:
    """
    # path1 = path1.reshape([-1, 45, 128])
    temp_importance = torch.Tensor().to(device)
    for place in path1:
        try:
            temp_importance = torch.cat([temp_importance, total_place_emb_importance[
                torch.nonzero(torch.eq(total_place_emb_importance[:, :-1], place).all(dim=1)).item()][-1].unsqueeze(
                dim=0)])
        except ValueError:
            temp_importance = torch.cat([temp_importance, torch.Tensor([0]).to(device)])
            count += 1
            # print(place)
    a = 0
    path1_emb_importance = torch.cat([path1, temp_importance.unsqueeze(dim=1)], dim=1)
    path1_emb_importance = path1_emb_importance.reshape([-1, 45, 129])

    path1 = path1.reshape([-1, 45, 128])
    val_place = torch.nonzero(path1.sum(dim=2)).tolist()
    val_len_dic = {}
    for i in range(len(val_place)):
        if val_place[i][0] not in val_len_dic.keys():
            val_len_dic[val_place[i][0]] = 1
        else:
            val_len_dic[val_place[i][0]] += 1
    for i in range(len(path1)):
        mask_c = val_len_dic[i]
        mask = random.sample(range(0, mask_c), max(1, round(mask_c * mask_ratio)))
        # indice = torch.topk(path1_emb_importance[i, :mask_c + 1, -1], max(1, int(mask_c * 0.5)), largest=False)[
        #     1].tolist()
        for m in mask:
            path1[i, m, :] = torch.zeros(128).to(device)
    return path1


def test_with_filter(dataloader, LP_model, reindexed_case_tensor_padded, config, topk, total_place_emb_importance,
                     mask_ratio):
    """
    filter place embeddings according to the place importance in test dataset
    :param dataloader:
    :param LP_model:
    :param reindexed_case_tensor_padded:
    :param config:
    :param topk:
    :param total_place_emb_importance:
    :param mask_ratio:
    :return:
    """
    device = torch.device('cuda:{}'.format(config.gpu)) if torch.cuda.is_available() else 'cpu'
    # classification_model.eval()
    total_place_emb_importance = total_place_emb_importance.to(device)
    roc, prt = {}, {}
    y, y_hat = [], []
    ratio_sum = 0
    count = 0
    LP_model.eval()
    # all_importance = torch.Tensor()
    count = 0
    for batch_x in dataloader:
        path1, path2, y_label = get_emb_label(reindexed_case_tensor_padded, batch_x, device)

        path1 = path1.reshape([-1, 128])
        path2 = path2.reshape([-1, 128])

        path1 = filter_tensor(total_place_emb_importance, path1, device, topk, count, mask_ratio)
        path2 = filter_tensor(total_place_emb_importance, path2, device, topk, count, mask_ratio)

        with torch.no_grad():
            path1 = path1.reshape([128, 45, 128]).permute(1, 0, 2)
            path2 = path2.reshape([128, 45, 128]).permute(1, 0, 2)

            y_hat_out, path1_emb, path2_emb, _, _ = LP_model(path1, path2)
            y_hat_out = y_hat_out.squeeze()
            y_hat += y_hat_out.cpu()
            y += y_label.cpu()
    roc['fpr'], roc['tpr'], roc['thresholds'] = metrics.roc_curve(y, y_hat, pos_label=1)
    prt['precision'], prt['recall'], prt['thresholds'] = metrics.precision_recall_curve(y, y_hat)
    auc = metrics.auc(roc['fpr'], roc['tpr'])

    return auc


def test_with_filter_random(dataloader, LP_model, reindexed_case_tensor_padded, config, topk,
                            total_place_emb_importance, mask_ratio):
    """
    filter place embeddings randomly in test dataset
    :param dataloader:
    :param LP_model:
    :param reindexed_case_tensor_padded:
    :param config:
    :param topk:
    :param total_place_emb_importance:
    :param mask_ratio:
    :return:
    """
    device = torch.device('cuda:{}'.format(config.gpu)) if torch.cuda.is_available() else 'cpu'

    total_place_emb_importance = total_place_emb_importance.to(device)
    roc, prt = {}, {}
    y, y_hat = [], []

    LP_model.eval()

    count = 0
    for batch_x in dataloader:
        path1, path2, y_label = get_emb_label(reindexed_case_tensor_padded, batch_x, device)

        path1 = path1.reshape([-1, 128])
        path2 = path2.reshape([-1, 128])

        path1 = filter_tensor_random(total_place_emb_importance, path1, device, topk, count, mask_ratio)
        path2 = filter_tensor_random(total_place_emb_importance, path2, device, topk, count, mask_ratio)

        with torch.no_grad():
            path1 = path1.reshape([128, 45, 128]).permute(1, 0, 2)
            path2 = path2.reshape([128, 45, 128]).permute(1, 0, 2)

            y_hat_out, path1_emb, path2_emb, _, _ = LP_model(path1, path2)
            y_hat_out = y_hat_out.squeeze()
            y_hat += y_hat_out.cpu()
            y += y_label.cpu()
    roc['fpr'], roc['tpr'], roc['thresholds'] = metrics.roc_curve(y, y_hat, pos_label=1)
    prt['precision'], prt['recall'], prt['thresholds'] = metrics.precision_recall_curve(y, y_hat)
    auc = metrics.auc(roc['fpr'], roc['tpr'])

    return auc


def get_data(config):
    """
    load data
    :param config:
    :return:
    """
    reindexed_sp_net, old2new_case_id, reindexed_case_tensor_padded = get_case_tensor('hprl_carl')
    train_dataloader, test_dataloader, all_dataloader = get_dataloader(config, reindexed_sp_net, old2new_case_id)

    place_train_dataset, place_test_dataset = torch.utils.data.random_split(train_dataloader.dataset,
                                                                            [len(train_dataloader.dataset) // 2,
                                                                             len(train_dataloader.dataset) - len(
                                                                                 train_dataloader.dataset) // 2])

    place_train_dataloader = torch.utils.data.DataLoader(place_train_dataset, batch_size=config.batch_size,
                                                         shuffle=True, drop_last=True)
    place_test_dataloader = torch.utils.data.DataLoader(place_test_dataset, batch_size=config.batch_size, shuffle=True,
                                                        drop_last=True)

    return place_train_dataloader, reindexed_case_tensor_padded, place_test_dataloader, train_dataloader, test_dataloader, all_dataloader


def min_loc(config):
    """
    mask places with different ratio and strategies
    :param config: parser object
    :return:
    """

    _, reindexed_case_tensor_padded, _, train_dataloader, test_dataloader, all_dataloader = get_data(
        config)


    LP_model = train(train_dataloader, config, reindexed_case_tensor_padded)

    LP_auc = test(test_dataloader, config, LP_model, reindexed_case_tensor_padded)
    print('LP auc {}'.format(LP_auc))

    total_place_emb, total_place_importance, total_place_emb_importance = fileter_place(all_dataloader,
                                                                                                   config,
                                                                                                   LP_model,
                                                                                                   reindexed_case_tensor_padded)
    min_auc = []
    random_auc = []
    for topk in range(1, 6):
        filter_auc = test_with_filter(test_dataloader, LP_model, reindexed_case_tensor_padded, config, topk,
                                      total_place_emb_importance, topk / 10.0)
        print('filter min:{}0%, test_with_filter auc {} '.format(topk, filter_auc))
        min_auc.append(filter_auc)
        filter_auc = test_with_filter_random(test_dataloader, LP_model, reindexed_case_tensor_padded, config, topk,
                                             total_place_emb_importance, topk / 10.0)
        print('filter random:{}0%, test_with_filter auc {} '.format(topk, filter_auc))
        random_auc.append(filter_auc)
    save_pickle((min_auc, random_auc), "saved/results/mask_ratio_compare.pickle")
    cl_loss = 0
    return LP_auc, cl_loss, filter_auc, LP_model, total_place_emb_importance
    # return LP_auc, cl_auc, None,LP_model,place_classification_model


def get_addr_importance(total_place_emb_importance, config):
    device = torch.device('cuda:{}'.format(config.gpu) if torch.cuda.is_available() else 'cpu')

    addr2emb = load_file('../represent_learning/hprl_carl_poi_embedding.pickle')

    addr2idx = {}
    idx2addr = {}
    count = 0

    for key, val in addr2emb.items():
        val = np.append(val, count)
        addr2emb[key] = torch.tensor(val, dtype=torch.double)

        addr2idx[key] = count
        idx2addr[count] = key
        count += 1

    addr2importance = {}
    cc = 0
    with torch.no_grad():
        for key, val in tqdm(addr2emb.items()):
            idx = int(val[-1])
            try:
                importance = total_place_emb_importance[
                    torch.nonzero(torch.eq(total_place_emb_importance[:, :-1], val[:-1]).all(dim=1)).item()][
                    -1].unsqueeze(dim=0)
                addr = idx2addr[idx]
                addr2importance[addr] = importance.item()
            except ValueError:
                importance = random.random() / 2.0
                addr = idx2addr[idx]
                addr2importance[addr] = importance
                cc += 1
    # save_pickle(addr2importance,'saved/addr2importance.pickle')
    # print('addr2importance saved')
    # print('cc:{}'.format(cc))


def main(config):
    setup_seed(10)
    _, _, _, _, total_place_emb_importance = min_loc(config)
    get_addr_importance(total_place_emb_importance, config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='att_LP', help='model name ')
    parser.add_argument('--lr_LP', type=float, default=18e-5, help='learning rate')
    parser.add_argument('--lr_path', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--ratio', type=float, default=1.0, help='ratio of neg samples and pos samples')
    parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs')
    parser.add_argument("--batch_size", type=int, default=128, help="LP batch size")
    parser.add_argument("--model_save_dir", type=str, default="saved/model/MLP", help="save dir for model")
    parser.add_argument("--HEAD", type=int, default=0, help="")
    parser.add_argument("--lbd", type=int, default=4)
    parser.add_argument('--path_len', type=int, default=45, help="max path length")
    parser.add_argument("--TAIL", type=int, default=1, help="")
    parser.add_argument("--LABEL", type=int, default=2, help="")
    parser.add_argument("--upperboundratio_10time", type=int, default=2,
                        help="upperboundratio_10time for ratio analysis")
    parser.add_argument("--sample_type", type=str, default='random',
                        help="sample type for pos and neg case pairs")
    parser.add_argument("--gpu", type=int, default=4, help="gpu device")
    parser.add_argument("--proportion", type=float, default=1.0, help="proportion of the total data for train and test")
    parser.add_argument("--importance_threshold", type=float, default=0.2, help="place importance threshold")
    parser.add_argument("--classification_num_epoch", type=int, default=100, help="number for classification epochs")
    parser.add_argument("--classification_lr", type=float, default=1e-4, help='classification learning rate')
    parser.add_argument("--epo", type=int, default=-1)
    config = parser.parse_args(args=[])
    main(config)
