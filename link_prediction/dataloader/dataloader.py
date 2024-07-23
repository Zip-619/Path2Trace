'''
Author: Zip
Date: 2021-11-23 12:07:00
LastEditors: Zip
LastEditTime: 2021-12-05 22:13:24
Description: dataloader for RNN,MLP
'''
import pickle
import sys

from torch.utils.data import Dataset

sys.path.append('')
from utils import generate_pos_neg_samples
import torch
from torch.utils.data import DataLoader
import os
from bidict import bidict


class path_data(Dataset):
    def __init__(self, data) -> None:
        # super().__init__(data)
        self.data = data

    # def __getitem__(self, index) :
    #     return super().__getitem__(index)

    def __getitem__(self, index):
        # return super().__getitem__(index)
        return self.data[index]

    def __len__(self):
        return len(self.data)

def get_dataloader(config, net, old2new_case_id,comp=False,idx_pair=[]):
    """
    get train and test data loader
    :param ratieo:
    :param batch_size:
    :param net:
    :return:
    """

    batch_size = config.batch_size
    sample_type = config.sample_type

    if os.path.exists('dataset/case_idx_pair/all_idx_pair_{}_{}.pickle'.format(str(config.lbd), sample_type)):
        print('load idx pair from exits data')
        with open('dataset/case_idx_pair/all_idx_pair_{}_{}.pickle'.format(str(config.lbd), sample_type), 'rb') as file:
            all_idx_pair = pickle.load(file=file)
            file.close()
    else:
        print('generate data for lambda:{}'.format(config.lbd))
        all_idx_pair = generate_pos_neg_samples(config,net,old2new_case_id)

    if os.path.exists('dataset/case_idx_pair/train_dataloader_{}_{}.pickle'.format(str(config.lbd), sample_type)):
        print('load dataloader from exits data')
        with open('dataset/case_idx_pair/train_dataloader_{}_{}.pickle'.format(str(config.lbd), sample_type), 'rb') as file:
            train_dataloader = pickle.load(file=file)
            file.close()
        with open('dataset/case_idx_pair/test_dataloader_{}_{}.pickle'.format(str(config.lbd), sample_type),
                  'rb') as file:
            test_dataloader = pickle.load(file=file)
            file.close()
        with open('dataset/case_idx_pair/all_dataloader_{}_{}.pickle'.format(str(config.lbd), sample_type),
                  'rb') as file:
            all_dataloader = pickle.load(file=file)
            file.close()
    else:
        data = path_data(all_idx_pair)
        train_size = int(0.8 * len(data))
        test_size = len(data) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(data, [train_size, test_size])
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        with open('dataset/case_idx_pair/train_dataloader_{}_{}.pickle'.format(str(config.lbd), sample_type), 'wb') as file:
            pickle.dump(train_dataloader,file,protocol=2)
            file.close()
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True,drop_last=True)
        with open('dataset/case_idx_pair/test_dataloader_{}_{}.pickle'.format(str(config.lbd), sample_type), 'wb') as file:
            pickle.dump(test_dataloader,file,protocol=2)
            file.close()
        all_dataloader = DataLoader(data,batch_size=config.batch_size,shuffle=True,drop_last=True)
        with open('dataset/case_idx_pair/all_dataloader_{}_{}.pickle'.format(str(config.lbd), sample_type), 'wb') as file:
            pickle.dump(all_dataloader,file,protocol=2)
            file.close()

    return train_dataloader, test_dataloader,all_dataloader




