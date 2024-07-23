"""
View more, visit my tutorial page: https://mofanpy.com/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou

Dependencies:
torch: 0.4
matplotlib
numpy
"""
import os

import numpy as np
import torch
# from torch._C import float32
import torch.nn as nn
import torch.utils.data as Data

# torch.manual_seed(1)    # reproducible

# Hyper Parameters
EPOCH = 20
BATCH_SIZE = 2
LR = 0.000005         # learning rate
# DOWNLOAD_MNIST = True
N_TEST_IMG = 5

data_path = '../pengpai/city_graph/data/'
embedding_path = os.path.join(data_path,'embedding/move_in/2021-01-19_70/weighted_1.emb')
total_embedding = np.fromfile(embedding_path,dtype=np.float32).reshape(-1,64)
# indice = range(len)
train_data = torch.tensor(list(total_embedding)[0:int(0.8*len(total_embedding))]).float()
# X_train = random.sample(list(total_embedding.values()),int(0.8*len(total_embedding)))
test_data = torch.tensor(list(total_embedding)[int(0.8*len(total_embedding)):len(total_embedding)]).float()

# Data Loader for easy mini-batch return in training, the image batch shape will be (50, 1, 28, 28)
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 96),
            nn.Tanh(),
            nn.Linear(96,64),
            nn.Tanh(),
            nn.Linear(64, 48),
            nn.Tanh(),
            nn.Linear(48, 32),
        )
        self.decoder = nn.Sequential(
            # nn.
            nn.Linear(32, 48),
            nn.Tanh(),
            nn.Linear(48, 64),
            nn.Tanh(),
            nn.Linear(64, 96),
            nn.Tanh(),
            nn.Linear(96, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


autoencoder = AutoEncoder()

optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LR)
loss_func = nn.MSELoss()


for epoch in range(EPOCH):
    for step, (x) in enumerate(train_loader):
        b_x = x.view(-1, 64)  # batch x, shape (batch, 28*28)
        b_y = x.view(-1, 64)   # batch y, shape (batch, 28*28)

        encoded, decoded = autoencoder(b_x)

        loss = loss_func(decoded, b_y)      # mean square error
        optimizer.zero_grad()               # clear gradients for this training step
        loss.backward()                     # backpropagation, compute gradients
        optimizer.step()                    # apply gradients

        if step % 100 == 0:
            # print(b_y)
            # print(encoded)
            # loss_func
            print('Epoch: ', epoch, '| train loss: %.8f' % loss.data.numpy())
            

print('####################################')

test_loader = Data.DataLoader(dataset = test_data,batch_size = BATCH_SIZE,shuffle=True)
for step,(x) in enumerate(test_loader):
    encoded, decoded = autoencoder(x)
    test_loss = loss_func(decoded,x)
    # print('decoded:{}'.format(decoded))
    # print('data:{}'.format(x))
    print('step {} test loss:{}'.format(step,test_loss))