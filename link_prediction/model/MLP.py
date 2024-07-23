'''
Author: Zip
Date: 2021-11-24 15:06:05
LastEditors: Zip
LastEditTime: 2021-11-25 14:50:41
Description: MLP model
'''
import torch
import torch.nn as nn


def init_noramal(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.01)

# class place_model(nn.Module):
#     def __init__(self):
#         super(place_model, self).__init__()
#         self.net = nn.Sequential(nn.Linear(128, 64),nn.Sigmoid(),nn.Linear(64, 32),nn.Sigmoid(),nn.Linear(32, 8),nn.Sigmoid(),nn.Linear(8, 1),nn.Sigmoid(),)
#
#     def forward(self,place_emb):
#         return self.net(place_emb)
#
# class path(nn.Module):
#     def __init__(self):
#         super(path, self).__init__()
#         self.net = nn.Sequential(
#             nn.Linear(5850, 200),
#             nn.Sigmoid(),
#             nn.Linear(200, 100),
#             nn.Sigmoid(),
#             nn.Linear(100, 50),
#             nn.Sigmoid(),
#             # nn.TransformerEncoderLayer(100, 10),
#             # nn.Linear(50,45),
#             # nn.Sigmoid(),
#         )
#         # self.s_a = nn.MultiheadAttention(embed_dim=130,num_heads=1)
#
#         # self.net.apply(init_noramal)
#         # self.trans = nn.TransformerEncoderLayer()
#     def forward(self, rawpath):
#         # return self.s_a(rawpath,rawpath,rawpath)
#         # att_path,weight = self.s_a(rawpath,rawpath,rawpath)
#         att_path = att_path.permute(1,0,2).reshape([128,-1])
#         return self.net(att_path),weight

class MLP_model(nn.Module):
    def __init__(self):
        super().__init__()
        # self.path_model_list = [place_model() for _ in range(90)]
        # self.path_model_list = torch.nn.ModuleList(self.path_model_list)
        self.path1 = nn.Sequential(
            nn.Linear(5760, 2000),
            nn.Sigmoid(),
            nn.Linear(2000, 1000),
            nn.Sigmoid(),
            nn.Linear(1000, 100),
            nn.Sigmoid(),
            nn.Linear(100, 45),
            nn.Sigmoid(),
        )
        self.path2 = nn.Sequential(
            nn.Linear(5760, 2000),
            nn.Sigmoid(),
            nn.Linear(2000, 1000),
            nn.Sigmoid(),
            nn.Linear(1000, 100),
            nn.Sigmoid(),
            nn.Linear(100, 45),
            nn.Sigmoid(),
        )
        self.MLP = nn.Sequential(
            nn.Linear(90, 128),
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
        # self.path1.apply(init_noramal)
        # self.path2.apply(init_noramal)
        self.MLP.apply(init_noramal)

    def forward(self, raw_path1, raw_path2):
        # device = torch.device("cuda:{}".format('1') if torch.cuda.is_available() else 'cpu')
        # paths_emb = torch.autograd.Variable(torch.tensor(torch.zeros(1,90))).to(device)

        path1_emb = self.path1(raw_path1)
        path2_emb = self.path2(raw_path2)
        paths_emb = torch.cat([path1_emb,path2_emb],dim=1)
        # for i,emb in enumerate(torch.split(raw_path1,128)):
        #     # print(emb.device)
        #     out = self.path_model_list[i](emb)
        #     paths_emb.data[0][i] = out
        # for i,emb in enumerate(torch.split(raw_path2,128)):
        #     out = self.path_model_list[i](emb)
        #     paths_emb.data[0][i+45] = out

        classfication_out = self.MLP(paths_emb)

        return classfication_out
