import argparse

try:
    from DNE.embedding import *
except ImportError:
    from embedding import *
try:
    from DNE.loading import *
except ImportError:
    from loading import *
import os
import random


def parse_args():
    parser = argparse.ArgumentParser(description="Run Verse.")

    parser.add_argument('--input', nargs='?', default='none',
                        help='Input graph path')

    parser.add_argument('--output', nargs='?', default='none',
                        help='Embeddings path')

    parser.add_argument('--base', nargs='?', default='none',
                        help='path of folder')

    parser.add_argument('--dimensions', type=int, default=64,
                        help='Number of dimensions. Default is 128.')

    parser.add_argument('--echos', type=int, default=12500,
                        help='Iterator number')

    parser.add_argument('--threads', type=int, default=4,
                        help='Threads number')

    parser.add_argument('--nsamples', type=int, default=3,
                        help='The number of negative samples')

    parser.add_argument('--jrate', type=float, default=0.85,
                        help='Jump rate')

    parser.add_argument('--test', nargs='?', default='none',
                        help='test file')

    parser.add_argument('--model', nargs='?', default='multi',
                        help='which model to use')

    return parser.parse_args()


# g++ -std=c++11 -march=native -fopenmp -Ofast -o verse-asymmetric verse-asymmetric.cpp #用来编译
# python3 DNE-multi --base ../data/LP_matrix_2014_70 --input all.train
if __name__ == '__main__':
    is_evalute = True  # 表示是否需要测试
    args = parse_args()

    args.test = '../data/LP_matrix_2015_50/18.test'

    base = args.base
    if base == 'none':
        base = '../data/LP_matrix_2015_50'

    if args.input == 'none':
        args.input = base + '/all.train'
    else:
        args.input = base + '/' + args.input

    base_folder = 'Parameter_Sensitive_emb_MDNE/' + base.split('/')[-1]

    if args.output == 'none':
        if not os.path.exists(base_folder):
            os.makedirs(base_folder)
        args.output = base_folder + '/' + args.model + '_' + args.input.split('/')[-1].split('.')[0] + '_dim_' + str(
            args.dimensions * 2) + '_nsample_' + str(args.nsamples) + '_jrate' + str(args.jrate) + '.emb'

    mid_path = base + '/' + args.input.split('/')[-1].split('.')[0] + '_' + args.model + '_' + str(
        random.randint(1, 1000000)) + '.bcsr'
    print('input = ', args.input, '. output = ', args.output)

    process(args.input, mid_path)  # 从训练集中生成得到中间bcsr文件

    command = 'src/dne-multi -input ' + mid_path + ' -output ' + args.output + ' -dim ' + str(
        args.dimensions) + ' -alpha ' + str(args.jrate) + ' -threads ' + str(args.threads) + ' -nsamples ' + str(
        args.nsamples) + ' -steps ' + str(args.echos)

    print(command)
    os.system(command)
    os.remove(mid_path)
    print('finish training, all embedding is saved on ', args.output)

    if is_evalute:
        s_path = args.output + '_s'
        t_path = args.output + '_t'
        l_path = args.output + '_l'
        node2id, title2id = get_word2id_from_file(args.input)
        s_model = get_embedding_from_path(s_path, args.dimensions, node2id)
        t_model = get_embedding_from_path(t_path, args.dimensions, node2id)
        l_model = get_embedding_from_path(l_path, args.dimensions, title2id)

    # 在G的转置上训练
    ########################################################################################################################
    print('<<<<Now we are training at GT>>>>')
    if '.train' in args.input:  # 此时是链接预测的任务
        args.input = args.input.split('.train')[0] + '_T.train'
    else:
        args.input = args.input + '_T'
    mid_path = base + '/' + args.input.split('/')[-1].split('.')[0] + '_' + args.model + '_' + str(
        random.randint(1, 10000)) + '.bcsr'
    args.output = base_folder + '/' + args.model + '_' + args.input.split('/')[-1].split('.')[0] + '_dim_' + str(
        args.dimensions * 2) + '_nsample_' + str(args.nsamples) + '_jrate' + str(args.jrate) + '.emb'
    print('input = ', args.input, '. output is = ', args.output)

    process(args.input, mid_path)  # 从训练集中生成得到中间bcsr文件

    command = 'src/dne-multi -input ' + mid_path + ' -output ' + args.output + ' -dim ' + str(
        args.dimensions) + ' -alpha ' + str(args.jrate) + ' -threads ' + str(args.threads) + ' -nsamples ' + str(
        args.nsamples) + ' -steps ' + str(args.echos)

    print(command)
    os.system(command)
    os.remove(mid_path)
    print('finish training, all embedding is saved on ', args.output)

    if is_evalute:
        T_s_path = args.output + '_s'
        T_t_path = args.output + '_t'
        T_l_path = args.output + '_l'
        node2id, title2id = get_word2id_from_file(args.input)
        T_s_model = get_embedding_from_path(T_s_path, args.dimensions, node2id)
        T_t_model = get_embedding_from_path(T_t_path, args.dimensions, node2id)
        T_l_model = get_embedding_from_path(T_l_path, args.dimensions, title2id)

        # if args.test != 'none':
        #     auc = verify_multi_G_GT(s_model, t_model, l_model, T_s_model, T_t_model, T_l_model, args.test)
        #     f = open('ps_mdne_result', 'a')
        #     f.write(args.model +'; '+ base +'; dim = ' + str(args.dimensions * 2) + '; nsample = ' + str(args.nsamples) + '; rate = ' + str(
        #             args.jrate) + '; 最终结果是：' + str(auc) + '\n')
        #     f.close()
        #     print(args.model +'\t'+args.base + '\t'+args.test + '\tdim = ' + str(args.dimensions * 2) + '\tnsample = ' + str(args.nsamples) + '\trate = ' + str(
        #         args.jrate) + '\t最终结果是:\t' + str(auc))
        # else:
        type_list = [str(i) for i in title2id.keys()]
        dic = {}
        for t in type_list:
            test_path = base + '/' + t + '.test'
            tmp_train = base + '/' + t + '.train'
            # auc = verify_multi_G_GT(s_model, t_model, l_model, T_s_model, T_t_model, T_l_model, test_path)
            auc = verify_strict_multi_G_GT(s_model, t_model, l_model, T_s_model, T_t_model, T_l_model, tmp_train,
                                           test_path)
            dic[t] = auc
        res = list()
        for t in dic:
            line = str(t) + ':' + str(dic[t])
            res.append(line)
        f = open('ps_mdne/ps_mdne_result_' + str(args.dimensions * 2), 'a')
        f.write(args.model + '; ' + base + '; dim = ' + str(args.dimensions * 2) + '; nsample = ' + str(
            args.nsamples) + '; rate = ' + str(
            args.jrate) + ';' + '最终结果是：' + ';'.join(res) + '\n')
        print('multi 最终结果 ：', args.input)
        for k in dic:
            print(k, '\t', dic[k])
        f.close()
