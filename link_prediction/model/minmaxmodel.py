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
    def __init__(self, place_emb_dim):
        super(path, self).__init__()
        self.s_a = nn.MultiheadAttention(embed_dim=place_emb_dim,num_heads=1)

    def forward(self, rawpath):
        att_path,weight = self.s_a(rawpath,rawpath,rawpath)
        att_path = att_path.permute(1,0,2).reshape([128,-1])
        return att_path,weight

class LP(nn.Module):
    def __init__(self, place_emb_dim,path_len):
        super(LP, self).__init__()
        self.LP = nn.Sequential(
            nn.Linear(place_emb_dim*path_len*2, 2000),
            nn.Sigmoid(),
            nn.Linear(2000, 1000),
            nn.Sigmoid(),
            nn.Linear(1000, 100),
            nn.Sigmoid(),
            nn.Sigmoid(),
            nn.Linear(100, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        self.LP.apply(init_noramal)
        self.path1 = path(place_emb_dim)
        self.path2 = path(place_emb_dim)

    def forward(self, p1, p2):
        path1_emb,path1_weight = self.path1(p1)
        path2_emb,path2_weight = self.path2(p2)
        double_path = torch.cat((path1_emb, path2_emb), 1)
        return self.LP(double_path), path1_emb, path2_emb,path1_weight,path2_weight


class place_classification(nn.Module):
    def __init__(self, input_dim):
        super(place_classification, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.Sigmoid(),
            nn.Linear(256, 128),
            nn.Sigmoid(),
            nn.Linear(128, 64),
            nn.Sigmoid(),
            nn.Linear(64, 16),
            nn.Sigmoid(),
            nn.Linear(16, 1),
        )

    def forward(self, place_emb):
        return self.net(place_emb)
