import time

slient = False

global_lr = 0.0025
n_epochs = 1000
n_hidden = 128
n_samples = 3
ppralpha = 0.85
total_steps = 0
step = 0

nv = 0
ne = 0

# w0 赋值
# w1 赋值

SIGMOID_BOUND = 6.0
sigmoid_table_size = 1024
sigmoid_table = []
SIGMOID_RESOLUTION = sigmoid_table_size/(SIGMOID_BOUND*2.0)

# rng_seed 

# 随机数函数

import numpy as np

def init_sigmoid_table():
    global sigmoid_table_size,SIGMOID_BOUND,sigmoid_table
    for i in range(sigmoid_table_size):
        x = 2*SIGMOID_BOUND*i/sigmoid_table_size-SIGMOID_BOUND
        sigmoid_table.append(1/(1+np.exp(-x)))
def fast_sigmoid(x):
    if(x>SIGMOID_BOUND):
        return 1
    elif(x<-SIGMOID_BOUND):
        return 0
    k = int((x+SIGMOID_BOUND)*SIGMOID_RESOLUTION)
    return sigmoid_table[k]

# sigmoid 函数
def sigmoid(x):
    return 1/(1+np.exp(-x))


def irand(min,max):
    
    return random.randint(min,max)

def irand(x):
    return random.randint(0,x-1)

def drand():
    return random.random()

import networkx as nx
def sample_rw(node):
    '''
    在network中从node开始,随机返回一个若干跳的邻居
    '''
    global net 
    next_node = node
    while(drand()<ppralpha):
        neighbors = list(net.neighbors(node))
        if (len(neighbors)==0):
            return node
        edges = list((node,neb) for neb in neighbors)
        probs = list()
        sum = 0
        for nb in neighbors:
            sum += net[node][nb]['weight'] 
        for nb in neighbors:
            probs.append(net[node][nb]['weight']/sum)
        next_node = np.random.choice(list(net.neighbors(node)),1,p = probs)
    return next_node

def update(n1,n2,label,bias): # 应该这里出了问题
#   对应于算法中的 UpdateBypair(u,v,k,D)
#   *w_s: n1的embedding?
#   *w_t: n2的embedding?
#   label: 1
#   bias: log(nv),节点个数的对数
    global ws,wt,global_lr
    score = -bias
    score += np.dot(np.squeeze(ws[n1]),np.squeeze(wt[n2]))

    # 试试 fastsigmoid 明天
    score = (label-fast_sigmoid(score))*global_lr # 这里用的是sigmoid 不是fast_sigmoid
    ws0 = ws[n1]
    wt0 = wt[n2]
    ws[n1] = wt0*score
    wt[n2] = ws0*score
    # return ws,wt

def Train(parser):
    nce_bias = np.log(nv)
    nce_bias_neg = np.log(nv/float(n_samples))
    last_ncount = 0
    ncount = 0
    global ws,wt,global_lr

    step = 0

    while(1):
        if (ncount-last_ncount>10000):
            diff = ncount-last_ncount
            step += diff
            if (step>total_steps):
                break
            if not parser.silent:
                print('\r Progress {}%'.format(step/(total_steps+1)*100))
            last_ncount = ncount
        n1 = irand(nv)
        n2 = sample_rw(n1)
        update(n1,n2,1,nce_bias)
        for i in range(n_samples):
            neg = irand(nv)
            update(n1,neg,0,nce_bias_neg)
        ncount += 1
    
    # return ws,wt

# 需要再次定义吗?
# def init_walker() 

import argparse
import random

def parse_arg():
    parser = argparse.ArgumentParser(description='run dne')
    parser.add_argument('-input', nargs='?',default= 'none',required=True,help="input graph")
    parser.add_argument('-output',nargs='?',default="none",required=True,help="embedding ouput path")
    parser.add_argument('-dim',type=int,default=n_hidden,help='embedding dim')
    parser.add_argument('-silent',type = bool,default=True,help='silent train')
    parser.add_argument('-nsamples', type=int, default=n_samples,help='The number of negative samples')
    parser.add_argument('-threads',type=int,default=4,help='number of thread')
    parser.add_argument('-lr',type=float,default=0.0025,help='learning rate')
    parser.add_argument('-alpha',type = float,default=ppralpha,help='jump rate')
    parser.add_argument('-init',type=bool,default=True,help='first time training')
    parser.add_argument('-steps',type= float,default=50, help='number of steps')
    
    return parser.parse_args()


if __name__=='__main__':
    parser = parse_arg()
    embedding_file = parser.output
    network_file = parser.input
    n_hidden = (int) (parser.dim)
    n_samples = parser.nsamples
    n_thread = parser.threads
    n_epochs = parser.steps
    global_lr = parser.lr
    ppralpha = parser.alpha
    init_sigmoid_table()
    if parser.init == True:
        net = nx.read_edgelist(network_file,nodetype = int,data=(('weight',float),),create_using = nx.DiGraph())
        nv = len(net.nodes())
        ne = len(net.edges())
        print('nv:{},ne:{}'.format(nv,ne))

        # 初始化为[-0.5,0.5]的均匀分布
        ws = np.random.random((nv,n_hidden))-0.5
        wt = np.random.random((nv,n_hidden))-0.5

        degrees = net.out_degree(net.nodes()) # 返回的是一个字典node:out_degree
        # print ('degrees:{}'.format(degrees))
        for node in net.nodes():
            sum = 0
            for nbr in net.neighbors(node):
                sum += net[node][nbr]['weight']
            if (degrees[node]==0):
                continue

            sum /= degrees[node]
            for nbr in net.neighbors(node):
                net[node][nbr]['weight'] /= sum

        total_steps = n_epochs*nv
        print('total steps (mil):{}'.format(total_steps/1000000.0))
        start = time.time()
        Train(parser)
        end = time.time()
        print('Caculations took:{} s to run'.format(end-start))

        print(ws)
        print(wt)

        ws.tofile(embedding_file+'_s')
        wt.tofile(embedding_file+'_t')
        # w = np.concatenate((ws,wt),axis=1)
        ws.tofile(embedding_file)






