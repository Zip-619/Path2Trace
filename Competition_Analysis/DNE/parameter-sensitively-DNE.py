import argparse

try:
    from DNE.embedding import *
except ImportError:
    from embedding import *
try:
    from DNE.convert import *
except ImportError:
    from convert import *
import random
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Run Verse.")

    parser.add_argument('--input', nargs='?', default='none',
                        help='Input graph path')

    parser.add_argument('--output', nargs='?', default='none',
                        help='Embeddings path')

    parser.add_argument('--base', nargs='?', default='none',
                        help='path of folder')

    parser.add_argument('--dimensions', type=int, default=64,
                        help='Number of dimensions. Default is 64.')

    parser.add_argument('--echos', type=int, default=50000,
                        help='Iterator number')

    parser.add_argument('--threads', type=int, default=4,
                        help='Threads number')

    parser.add_argument('--nsamples', type=int, default=3,
                        help='The number of negative samples')

    parser.add_argument('--jrate', type=float, default=0.85,
                        help='Jump rate')

    parser.add_argument('--test', nargs='?', default='none',
                        help='test file')

    parser.add_argument('--model', nargs='?', default='asymmetric',
                        help='which model to use')

    parser.add_argument('--type', nargs='?', default='18',
                        help='which network to use')

    return parser.parse_args()


# g++ -std=c++11 -march=native -fopenmp -Ofast -o verse-asymmetric verse-asymmetric.cpp #用来编译
# python3 verse.py --model asymmetric --type 7
# python3 DNE-model.py --base ../data/All_matrix_2014 --input 7_all --threads 1 # 对整个图训练
if __name__ == '__main__':
    is_evalute = True  # 表示是否需要测试
    args = parse_args()

    base = args.base
    if base == 'none':
        base = '../data/LP_matrix_2015_50'

    if args.input == 'none':
        args.input = base + '/' + args.type + '.train'
    else:
        args.input = base + '/' + args.input

    mid_path = base + '/' + args.input.split('/')[-1].split('.')[0] + '_' + args.model + '_' + str(
        random.randint(1, 1000000)) + '.bcsr'

    base_folder = 'Parameter_Sensitive_emb_DNE/' + base.split('/')[-1]

    if args.output == 'none':
        if not os.path.exists(base_folder):
            os.makedirs(base_folder)
        args.output = base_folder + '/' + args.model + '_' + args.input.split('/')[-1].split('.')[0] + '_dim_' + str(args.dimensions * 2) + '_nsample_' + str(args.nsamples) + '_jrate_' + str(args.jrate) + '.emb'
    print('input = ', args.input, '. output is = ', args.output)

    main(format='weighted_edgelist', matfile_variable_name='network', undirected=False, sep=None, input=args.input,
         output=mid_path)

    if args.model == 'weighted':  # 为每个节点训练一个向量
        command = 'src/dne-weighted -input ' + mid_path + ' -output ' + args.output + ' -dim ' + str(
            args.dimensions) + ' -alpha ' + str(args.jrate) + ' -threads ' + str(args.threads) + ' -nsamples ' + str(
            args.nsamples) + ' -steps ' + str(args.echos)
    elif args.model == 'asymmetric':  # 初始化时s和t向量都是[-0.5,0.5]的均匀分布
        command = 'src/dne-asymmetric -input ' + mid_path + ' -output ' + args.output + ' -dim ' + str(
            args.dimensions) + ' -alpha ' + str(args.jrate) + ' -threads ' + str(args.threads) + ' -nsamples ' + str(
            args.nsamples) + ' -steps ' + str(args.echos)
    else:
        print('model is not defined')
        exit()

    print(command)

    os.system(command)
    os.remove(mid_path)
    print('finish training, all embedding is saved on ', args.output)

    node2id, _ = get_word2id_from_file(args.input)
    if is_evalute:
        if args.test == 'none':
            args.test = base + '/' + args.type + '.test'
        s_path = args.output + '_s'
        t_path = args.output + '_t'
        s_model = get_embedding_from_path(s_path, args.dimensions, node2id)
        t_model = get_embedding_from_path(t_path, args.dimensions, node2id)

    # 在G的转置上训练
    ########################################################################################################################
    print('<<<<Now we are training at GT>>>>')
    if '.train' in args.input:  # 此时是链接预测的任务
        args.input = args.input.split('.train')[0] + '_T.train'
    else:
        args.input = args.input + '_T'
    mid_path = base + '/' + args.input.split('/')[-1].split('.')[0] + '_' + args.model + '_' + str(
        random.randint(1, 10000)) + '.bcsr'

    args.output = base_folder + '/' + args.model + '_' + args.input.split('/')[-1].split('.')[0] + '_dim_' + str(args.dimensions * 2) + '_nsample_' + str(args.nsamples) + '_jrate_' + str(args.jrate) + '.emb'

    print('input = ', args.input, '. output is = ', args.output)

    main(format='weighted_edgelist', matfile_variable_name='network', undirected=False, sep=None, input=args.input,
         output=mid_path)

    command = 'src/dne-asymmetric -input ' + mid_path + ' -output ' + args.output + ' -dim ' + str(
        args.dimensions) + ' -alpha ' + str(args.jrate) + ' -threads ' + str(args.threads) + ' -nsamples ' + str(
        args.nsamples) + ' -steps ' + str(args.echos)

    print(command)

    os.system(command)
    os.remove(mid_path)
    print('finish training, all embedding is saved on ', args.output)

    node2id, _ = get_word2id_from_file(args.input)
    if is_evalute:
        T_s_path = args.output + '_s'
        T_t_path = args.output + '_t'
        T_s_model = get_embedding_from_path(T_s_path, args.dimensions, node2id)
        T_t_model = get_embedding_from_path(T_t_path, args.dimensions, node2id)

        auc = verify_G_GT(s_model, t_model, T_s_model, T_t_model, args.test)
        print(args.model, ' dim = ', args.dimensions, ' nsample = ', args.nsamples, ' rate = ', args.jrate, ' 最终结果是：',auc)
        f = open('ps_dne/ps_dne_result_'+str(args.dimensions*2), 'a')
        f.write(args.model + '; ' + base + '; ' + args.test + '; dim = ' + str(args.dimensions * 2) + '; nsample = ' + str(
            args.nsamples) + '; rate = ' + str(args.jrate) + '; 最终结果是：' + str(auc) + '\n')
        f.close()
