'''
Author: Zip
Date: 2021-11-24 21:35:30
LastEditors: Please set LastEditors
LastEditTime: 2021-11-25 19:42:16
Description: loc model
'''

# class loc_model(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.path1 = nn.Sequential(
#             nn.Linear(90,200),
#             nn.Sigmoid(),
#             nn.Linear(200,100),
#             nn.Sigmoid(),
#             nn.Linear(100,50),
#         )
#         self.path2 = nn.Sequential(
#             nn.Linear(90,200),
#             nn.Sigmoid(),
#             nn.Linear(200,100),
#             nn.Sigmoid(),
#             nn.Linear(100,50),
#         )
#         self.MLP = nn.Sequential(
#             nn.Linear(100,128),
#             nn.Tanh(),
#             nn.Linear(128,64),
#             nn.Tanh(),
#             nn.Linear(64,32),
#             nn.Tanh(),
#             nn.Linear(32,16),
#             nn.Tanh(),
#             nn.Linear(16,1),
#             nn.Sigmoid()
#         )
#
#     def forward(self,raw_path1,raw_path2):
#         path1_emb = self.path1(raw_path1)
#         path2_emb = self.path2(raw_path2)
#         doubelpath = torch.cat((path1_emb,path2_emb),1)
#         classfication_out = self.MLP(doubelpath)
#         return classfication_out

import torch
import torch.nn as nn


def init_noramal(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.01)


class path_loss(nn.Module):
    def __init__(self):
        super(path_loss, self).__init__()

    def forward(self, input):
        sig = nn.Sigmoid()
        loss = sig(-torch.norm(input, 1))
        # loss = -torch.std(input)
        return loss



class path(nn.Module):
    def __init__(self):
        super(path, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(90, 200),
            nn.Sigmoid(),
            nn.Linear(200, 100),
            nn.Sigmoid(),
            nn.Linear(100, 50),
            nn.Sigmoid(),
            # nn.TransformerEncoderLayer(100, 10),
            # nn.Linear(50,45),
            # nn.Sigmoid(),
        )
        self.s_a = nn.MultiheadAttention(embed_dim=2,num_heads=1)

        # self.net.apply(init_noramal)
        # self.trans = nn.TransformerEncoderLayer()
    def forward(self, rawpath):
        # return self.s_a(rawpath,rawpath,rawpath)
        att_path,weight = self.s_a(rawpath,rawpath,rawpath)
        att_path = att_path.permute(1,0,2).reshape([128,-1])
        return att_path, weight


class loc_model(nn.Module):
    def __init__(self):
        super(loc_model, self).__init__()
        self.LP = nn.Sequential(
            nn.Linear(180, 128),
            # nn.Linear(128,128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            # nn.Linear(16,2), # 不用计算auc时的网络
            nn.Linear(16, 1),  # 要计算auc时的输出
            nn.Sigmoid()  # 要计算auc时的网络
        )
        self.LP.apply(init_noramal)
        self.path1 = path()
        self.path2 = path()

    def forward(self, p1, p2):
        path1_emb,path1_weight = self.path1(p1)
        path2_emb,path2_weight = self.path2(p2)
        double_path = torch.cat((path1_emb, path2_emb), 1)
        return self.LP(double_path), path1_emb, path2_emb,path1_weight,path2_weight

# <<<<<<< HEAD
#
# class place_classification(nn.Module):
#     def __init__(self, input_dim):
#         super(place_classification, self).__init__()
#         self.net = nn.Sequential(
#             nn.Linear(input_dim, 256),
#             nn.Sigmoid(),
#             nn.Linear(256, 128),
#             nn.Sigmoid(),
#             nn.Linear(128, 64),
#             nn.Sigmoid(),
#             nn.Linear(64, 16),
#             nn.Sigmoid(),
#             nn.Linear(16, 1),
#             # nn.Tanh()
#         )
#
#     def forward(self, place_emb):
#         return self.net(place_emb)
