# from torch.utils.tensorboard import SummaryWriter
import argparse
import os
import torch.utils.data
from dataloader.dataloader import get_dataloader
from model.loc_model import loc_model
from model.minmaxmodel import LP
import torch.nn as nn
from utils import *
import sys
sys.path.append('../data/pengpai')
from address import address

def train(train_dataloader, config, reindexed_case_tensor_padded):
    """ training process

    Args:
        train_dataloader: dataloader
        config: parser
        reindexed_case_tensor_padded: case representation

    Returns:
        pytorch model
    """
    batch_size = config.batch_size
    path_len = config.path_len
    place_emb_dim = int(reindexed_case_tensor_padded.shape[1] / path_len)
    device = torch.device('cuda:{}'.format(config.gpu) if torch.cuda.is_available() else 'cpu')

    LP_model = LP(place_emb_dim=place_emb_dim, path_len=path_len)
    if config.rl_name == 'loc':
        # If we use latitude and longitude as the representation of the location,
        # we need to reconstruct the network because of its low dimensionality,
        # and cannot use the original network structure directly.
        LP_model = loc_model()
    opt_LP = torch.optim.Adam(LP_model.parameters(), lr=config.lr_LP)
    LP_loss = nn.MSELoss()

    LP_model.to(device)

    epochs = tqdm(range(config.num_epochs), ncols=100)
    count = 0
    for _ in epochs:
        for batch_x in train_dataloader:
            path1, path2, y = get_emb_label(reindexed_case_tensor_padded, batch_x, device)

            opt_LP.zero_grad()
            path1 = path1.reshape([batch_size, path_len, place_emb_dim]).permute(1, 0, 2)
            path2 = path2.reshape([batch_size, path_len, place_emb_dim]).permute(1, 0, 2)
            # path1 shape torch.Size([45, 128, 128])
            # attn_output: (L, N, E)(L,N,E) where L is the target sequence length(45),
            # N is the batch size(128), E is the embedding dimension.
            # (N, L, E)(N,L,E) if batch_first is True.
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

    # save model
    torch.save(LP_model.state_dict(),
               '{}/{}_{}_model.pth'.format(config.model_save_dir, config.rl_name, config.num_epochs))

    return LP_model


def test(test_dataloader, config, LP_model, reindexed_case_tensor_padded):
    """

    Args:
        test_dataloader: pytorch dataloader
        config: parser object
        LP_model: model
        reindexed_case_tensor_padded: representation of cases

    Returns:
        results: tuple test results
    """
    y, y_hat = [], []
    batch_size = config.batch_size
    path_len = config.path_len
    place_emb_dim = int(reindexed_case_tensor_padded.shape[1] / path_len)
    device = torch.device("cuda:{}".format(config.gpu) if torch.cuda.is_available() else 'cpu')
    LP_model.to(device)
    for batch_x in test_dataloader:
        path1, path2, y_label = get_emb_label(reindexed_case_tensor_padded, batch_x, device)
        with torch.no_grad():
            path1 = path1.reshape([batch_size, path_len, place_emb_dim]).permute(1, 0, 2)
            path2 = path2.reshape([batch_size, path_len, place_emb_dim]).permute(1, 0, 2)
            y_hat_out, _, _, _, _ = LP_model(path1, path2)
            y_hat_out = y_hat_out.squeeze()
            y_label = y_label.float()
        y_hat += y_hat_out.cpu()
        y += y_label.cpu()

    # roc curve
    fpr, tpr, thresholds = metrics.roc_curve(y, y_hat, pos_label=1)
    # Find the best cutoff point based on the Jorden index
    opt_thd, opt_point = find_optimal_cutoff_roc(tpr, fpr, thresholds)
    # Determine the model output based on the best cutoff point
    y_b_hat = y_hat.copy()
    for i, val in enumerate(y_hat):
        if val > opt_thd:
            y_b_hat[i] = 1
        else:
            y_b_hat[i] = 0
    # calculate acc precision recall f1 auc
    acc = metrics.accuracy_score(y, y_b_hat)
    precision = metrics.precision_score(y, y_b_hat)
    recall = metrics.recall_score(y, y_b_hat)
    f1 = metrics.f1_score(y, y_b_hat)
    auc = metrics.roc_auc_score(y, y_hat)

    print('accurancy:{}'.format(metrics.accuracy_score(y, y_b_hat)))
    print('precision: {}'.format(metrics.precision_score(y, y_b_hat)))
    print('recall:{}'.format(metrics.recall_score(y, y_b_hat)))
    print('F1:{}'.format(metrics.f1_score(y, y_b_hat)))
    print('auc:{}'.format(metrics.roc_auc_score(y, y_hat)))

    return (acc, precision, recall, f1, auc, torch.tensor(y).tolist(), torch.tensor(y_hat).tolist(),torch.tensor(y_b_hat).tolist())


def get_data(config):
    """

    Load data according to the parameters in the config

    Args:
        config: parser object

    Returns:

        reindexed_case_tensor_padded：Representation of cases, aligned with 0 fill
        train_dataloader: subset of all_dataloader for training
        test_dataloader: subset of all_dataloader for testing
        all_dataloader: all cases pair

    """
    # 获得train_data_loader和test_dataloader

    # 根据参数中的不同表示学习方法加载不同的地点的表征学习结果，从而得到不同的病例的表征
    reindexed_sp_net, old2new_case_id, reindexed_case_tensor_padded = get_case_tensor(config.rl_name)
    # 根据病例的表征以及传播网络构建训练集和测试集
    train_dataloader, test_dataloader, all_dataloader = get_dataloader(config, reindexed_sp_net, old2new_case_id)
    # print('total sample numbers：{}'.format(len(all_dataloader.dataset)))

    # place_train_dataset, place_test_dataset for other tasks
    place_train_dataset, place_test_dataset = torch.utils.data.random_split(train_dataloader.dataset,
                                                                            [len(train_dataloader.dataset) // 2,
                                                                             len(train_dataloader.dataset) - len(
                                                                                 train_dataloader.dataset) // 2])
    place_train_dataloader = torch.utils.data.DataLoader(place_train_dataset, batch_size=config.batch_size,
                                                         shuffle=True, drop_last=True)
    place_test_dataloader = torch.utils.data.DataLoader(place_test_dataset, batch_size=config.batch_size, shuffle=True,
                                                        drop_last=True)

    return reindexed_case_tensor_padded, train_dataloader, test_dataloader, all_dataloader


def train_on_model(model_name_list):
    """
    Initialize different models and load different case representations according to the different model parameter names passed in
    Args:
        model_name_list: list[str] list of the RL model names

    Returns:
        None
    """

    save_path = config.res_save_path
    for rl_name in model_name_list:
        print('train on {}...'.format(rl_name))
        config.rl_name = rl_name
        # Load dataloader
        reindexed_case_tensor_padded, train_dataloader, test_dataloader, all_dataloader = get_data(
            config)
        LP_model = train(train_dataloader, config, reindexed_case_tensor_padded)
        test_res = test(test_dataloader, config, LP_model, reindexed_case_tensor_padded)
        save_pickle(obj=test_res, filepath=os.path.join(save_path, '{}.pickle'.format(rl_name)))


def main(config):
    """

    Entrance

    Args:
        config: Parser object

    Returns:
        None

    """

    # print(config)
    setup_seed(20)
    if config.run_model == 'results': # 运行baselines
        baselines = ['spacene','louvainne','deepwalk','node2vec','randne','boostne','sdne','gae','vgae']
        train_on_model(baselines)

    if config.run_model == 'ablation': # 消融实验
        baselines = ['hprl', 'loc','carl_loc']
        train_on_model(baselines)

    if config.run_model == 'train': # 基础模型训练
        reindexed_case_tensor_padded, train_dataloader, test_dataloader, all_dataloader = get_data(
            config)
        LP_model = train(train_dataloader, config, reindexed_case_tensor_padded)
        _ = test(test_dataloader, config, LP_model, reindexed_case_tensor_padded)

    if config.run_model == 'lr_lbd_sen': # lr和lambda的网格化搜索
        config.lr_LP = 1e-5
        lr = config.lr_LP
        res = load_file('saved/sen_res.pickle')
        # res = {}
        for i in range(0, 10):  # lr
            # res[i]={}
            config.lr_LP = lr * (2 ** i)
            if i not in res.keys():
                res[i] = {}
            for j in range(40, 41):  # lbd
                if j in res[i].keys():
                    continue
                print('train in epoch:{}'.format(i))
                print('learning rate: {}'.format(config.lr_LP))
                config.lbd = j
                print('lambda={}'.format(config.lbd))
                place_train_dataloader, reindexed_case_tensor_padded, place_test_dataloader, train_dataloader, test_dataloader, all_dataloader = get_data(
                    config)

                LP_model = train(place_train_dataloader, config, reindexed_case_tensor_padded)
                LP_auc = test(place_test_dataloader, config, LP_model, reindexed_case_tensor_padded)
                # print('LP auc {}'.format(LP_auc))
                # config.lr_LP += 1e-5
                res[i][j] = LP_auc
                save_pickle(res, 'saved/sen_res.pickle')

            # lr = config.lr_LP

        save_pickle(res, 'saved/sen_res.pickle')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--lr_LP', type=float, default=18e-5, help='learning rate')
    parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs')
    parser.add_argument("--batch_size", type=int, default=128, help="LP batch size")
    parser.add_argument("--model_save_dir", type=str, default="saved/model", help="save dir for model")
    parser.add_argument("--HEAD", type=int, default=0, help="")
    parser.add_argument("--TAIL", type=int, default=1, help="")
    parser.add_argument("--LABEL", type=int, default=2, help="")
    parser.add_argument("--upperboundratio_10time", type=int, default=2,
                        help="upperboundratio_10time for ratio analysis")

    parser.add_argument("--sample_type", type=str, default='random',
                        help="sample type for pos and neg case pairs")

    parser.add_argument("--gpu", type=int, default=1, help="gpu device")
    parser.add_argument("--proportion", type=float, default=1.0, help="proportion of the total data for train and test")

    parser.add_argument("--run_model", type=str, default='ablation',help="run options")
    parser.add_argument("--lbd", type=int, default=4,help="hyperparameter for sample positive and negative samples")
    parser.add_argument("--get_best", type=bool, default=False,help="whether run to get best results")
    parser.add_argument("--epo", type=int, default=-1,help="load epoch train results")
    parser.add_argument('--path_len', type=int, default=45, help="max path length")
    parser.add_argument("--num_gb_epoch", type=int, default=20, help="number of epoch for geting best model")
    parser.add_argument('--rl_name', type=str, default='hprl_carl', help="representation learning method name")
    parser.add_argument('--res_save_path', type=str, default='saved/results',help="results save path")
    config = parser.parse_args(args=[])
    main(config)
