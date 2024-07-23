class exp(object):
    def __init__(self, string):
        arr = string.split(',')
        self.title = arr[0]
        self.com = arr[1]
        self.s_time = arr[2].split('-')[0]
        self.e_time = arr[2].split('-')[1]

    def tostring(self):
        return self.title + ',' + self.com + ',' + self.s_time + '-' + self.e_time


def removeRepeat(line):
    """
    去掉职业生涯链中重复的经历
    """
    arr = line.strip().split('.')
    exps = list()
    for a in arr:
        e = exp(a)
        exps.append(e)
    res = list()
    for i in range(0, len(exps) - 1):
        if exps[i].title == exps[i + 1].title and exps[i].com == exps[i + 1].com:
            exps[i + 1].s_time = exps[i].s_time
            exps[i].e_time = exps[i + 1].e_time
        else:
            string = exps[i].tostring()
            res.append(string)
    res.append(exps[len(exps) - 1].tostring())
    return '.'.join(res)


target_com = set([str(i) for i in range(38824)])
target_title = set([str(i) for i in range(1, 27)])


def filterLine(line):
    """
    进行过滤
    """
    arr = line.strip().split('.')
    if len(arr) == 1:
        return ''
    exps = list()
    for a in arr:
        e = exp(a)
        exps.append(e)

    cur_com = list()  # 一行数据中目标公司对应的下标
    cur_set = set()  # 一行数据中目标公司的集合
    cur_title = list()  # 一行数据中目标岗位对应的下标
    for i in range(0, len(exps)):
        if exps[i].com in target_com:
            cur_com.append(i)
            cur_set.add(exps[i].com)
        if exps[i].title in target_title:
            cur_title.append(exps[i].title)
    if len(cur_set) <= 1 or len(cur_title) == 0:
        return ''
    f_index = cur_com[0]
    t_index = cur_com[-1]
    return '.'.join(arr[f_index:t_index + 1])


def format_data(input, output):
    f = open(output, 'w')
    for line in open(input):
        newline = removeRepeat(line)
        newline = filterLine(newline)
        if len(newline) > 0:
            f.write('%s\n'%newline)
    f.close()


def get_pure_link(input='format_chain_2014.txt', output='pure.txt', base='/Users/laughing/Desktop/newdata/'):
    """
    生成纯碎的链。一条职业生涯链中全部是同一个岗位
    """

    input = base + input
    output = base + output
    links = dict()
    for k in target_title:
        links[k] = 0
    f = open(output, 'w')
    for line in open(input):
        titles = set()
        arr = line.strip().split('.')
        for a in arr:
            titles.add(a.split(',')[0])
        if len(titles) == 1 and list(titles)[0] in target_title:
            f.write(line)
            links[list(titles)[0]] += 1
    f.close()
    print('title \t pure中链数')
    for k in links:
        print(k, '\t', links[k])


def get_relax_pure_link(input='format_chain_2014.txt', output='relax.txt', base='/Users/laughing/Desktop/newdata/'):
    """
    生成纯碎的链。一条职业生涯链中全部是同一个岗位
    """
    input = base + input
    output = base + output
    links = dict()
    for k in target_title:
        links[k] = 0
    f = open(output, 'w')
    for line in open(input):
        arr = line.strip().split('.')
        title_count = dict()
        for i in range(0, len(arr)):
            title = arr[i].split(',')[0]
            if title not in target_title:
                continue
            if title not in title_count:
                title_count[title] = 0
            title_count[title] += 1
        exps = list()
        for k in title_count:
            if title_count[k] >= round(len(arr) / 2):
                links[k] += 1
                for a in arr:
                    string = k + ',' + a.split(',')[1] + ',' + a.split(',')[2]
                    exps.append(string)
                f.write('.'.join(exps) + '\n')
                break
    f.close()
    print('title \t pure_relax中链数(包含)')
    for k in links:
        print(k, '\t', links[k])


def count_for_pure(path):
    """
    为pure统计不同title的链数
    :param path:
    :return:
    """
    links = dict()
    for line in open(path):
        arr = line.strip().split(' ')
        title = arr[0].split(',')[0]
        if title not in links:
            links[title] = 0
        links[title] += 1
    print('title \t pure中链数')
    for title in links:
        print(title, '\t', links[title])


def get_direct_matrix(path):
    """
    从链数据中获取矩阵,只有前后两个公司才叫跳转
    """
    matrix = dict()
    for title in target_title:
        matrix[title] = dict()
    for line in open(path):
        arr = line.strip().split('.')
        for i in range(0, len(arr) - 1):
            f_title = arr[i].split(',')[0]
            t_title = arr[i + 1].split(',')[0]
            f_com = arr[i].split(',')[1]
            t_com = arr[i + 1].split(',')[1]
            if f_title == t_title and f_com != t_com and f_com in target_com and t_com in target_com and f_title in target_title:
                hop = f_com + ' ' + t_com
                if hop not in matrix[f_title]:
                    matrix[f_title][hop] = 0
                matrix[f_title][hop] += 1
    print('title\t网络对应的边（只考虑直接情况）')
    for k in matrix:
        print(k, '\t', len(matrix[k]))
    return matrix


def get_matrix_between_duration(path, f_time, t_time):
    """
    得到某个时间段的转移矩阵，离职时间按照入职的时间计算，要求入职日期大于等于f_time，小于t_time
    :param path:
    :param f_time:
    :param t_time:
    :return:
    """
    matrix = dict()
    for title in target_title:
        matrix[title] = dict()
    for line in open(path):
        arr = line.strip().split('.')
        for i in range(0, len(arr)-1):
            f_title = arr[i].split(',')[0]
            t_title = arr[i + 1].split(',')[0]
            f_com = arr[i].split(',')[1]
            t_com = arr[i + 1].split(',')[1]
            left_year = int(arr[i + 1].split(',')[2].split('/')[0])
            if left_year < f_time or left_year > t_time:
                continue
            if f_title == t_title and f_com != t_com and f_com in target_com and t_com in target_com and f_title in target_title:
                hop = f_com + ' ' + t_com
                if hop not in matrix[f_title]:
                    matrix[f_title][hop] = 0
                matrix[f_title][hop] += 1
    print('title\t网络对应的边（只考虑直接情况）')
    for k in matrix:
        print(k, '\t', len(matrix[k]))
    return matrix



def get_indirect_matrix(path):
    """
       从链数据中获取矩阵,将链中非目标公司去掉，相邻两个公司构成一个跳转
    """
    matrix = dict()
    for title in target_title:
        matrix[title] = dict()
    for line in open(path):
        arr = line.strip().split('.')
        coms = list()
        for i in range(0, len(arr)):
            com = arr[i].split(',')[1]
            if com in target_com:
                coms.append(com)
        title = arr[0].split(',')[0]
        for i in range(0, len(coms) - 1):
            if coms[i] == coms[i + 1]:
                continue
            hop = coms[i] + ' ' + coms[i + 1]
            if hop not in matrix[title]:
                matrix[title][hop] = 0
            matrix[title][hop] += 1
    print('title\t网络对应的边(考虑间接情况)')
    for k in matrix:
        print(k, '\t', len(matrix[k]))
    return matrix


def getCommonNodes(matrix_path, title_list):
    title_nodes = dict()
    for title in title_list:
        title_nodes[title] = set()
    for line in open(matrix_path):
        arr = line.strip().split(' ')
        title = arr[0]
        if title in title_list:
            f_node = arr[1]
            t_node = arr[2]
            title_nodes[title].add(f_node)
            title_nodes[title].add(t_node)
    common_nodes = set()
    print('网络\t包含节点数')
    for title in title_list:
        print(title, '\t', len(title_nodes[title]))
    for node in title_nodes[title_list[0]]:
        flag = True
        for i in range(1, len(title_list)):
            if node not in title_nodes[title_list[i]]:
                flag = False
                break
        if flag == True:
            common_nodes.add(node)
    print(len(common_nodes))
    return common_nodes


def filterCommonNodes(matrix_path, title_list, suffix='2015'):
    """
    计算每个节点在matrix中每个网络上出现的次数
    :param matrix_path:
    :param title_list:
    :return:
    """
    title_list = set(title_list)
    node_title = dict()
    node_title_indegree = dict()
    node_title_outdegree = dict()
    for com in target_com:
        node_title[com] = dict()
        node_title_indegree[com] = dict()
        node_title_outdegree[com] = dict()
    for line in open(matrix_path):
        arr = line.strip().split(' ')
        title = arr[0]
        f_node = arr[1]
        t_node = arr[2]
        if title in title_list:
            if title not in node_title[f_node]:
                node_title[f_node][title] = 0
            if title not in node_title[t_node]:
                node_title[t_node][title] = 0
            if title not in node_title_outdegree[f_node]:
                node_title_outdegree[f_node][title] = 0
            if title not in node_title_indegree[t_node]:
                node_title_indegree[t_node][title] = 0
            node_title[f_node][title] += 1
            node_title[t_node][title] += 1
            node_title_indegree[t_node][title] += 1
            node_title_outdegree[f_node][title] += 1
    f = open('/Users/laughing/Desktop/newdata/node_title_count_'+suffix, 'w')
    f_out = open('/Users/laughing/Desktop/newdata/node_title_outdegree_'+suffix, 'w')
    f_in = open('/Users/laughing/Desktop/newdata/node_title_indegree_'+suffix, 'w')
    for node in node_title:
        f.write(node + ';' + '.'.join([k + ':' + str(node_title[node][k]) for k in node_title[node]])+'\n')
    f.close()
    for node in node_title_outdegree:
        f_out.write(node + ';' + '.'.join([k + ':' + str(node_title[node][k]) for k in node_title[node]])+'\n')
    f_out.close()
    for node in node_title_indegree:
        f_in.write(node + ';' + '.'.join([k + ':' + str(node_title[node][k]) for k in node_title[node]])+'\n')
    f_in.close()
    return node_title



def construct_matrix_base_nodes_titles(matrix_path, title_list, common_nodes, output):
    f = open(output, 'w')
    # common_nodes = getCommonNodes(matrix_path, title_list)
    edges = dict()
    for line in open(matrix_path):
        arr = line.strip().split(' ')
        title = arr[0]
        f_node = arr[1]
        t_node = arr[2]
        if title in title_list and f_node in common_nodes and t_node in common_nodes:
            f.write(line)
            if title not in edges:
                edges[title] = 0
            edges[title] += 1
    print('title', '\t边数')
    for k in edges:
        print(k, '\t', edges[k])
    f.close()


def read_node_title_count(path):
    node_title = dict()
    for line in open(path):
        node = line.strip().split(';')[0]
        node_title[node] = dict()
        words = line.strip().split(';')[1].split('.')
        if len(line.strip().split(';')[1]) == 0:
            continue
        for word in words:
            title = word.split(':')[0]
            count = int(word.split(':')[1])
            node_title[node][title] = count
    return node_title

"""
careerChain_2014.txt : 2014年以后所有的职业生涯链
format_chain_2014.txt : 对careerChain_2014进行去重，格式化
pure.txt : 一条职业生涯链中全部是同一个岗位
relax.txt : 一条链中某个岗位大于等于总共岗位的一半，就将所有的岗位设置成这个岗位
pure_matrix.txt : pure.txt构成的matrix
node_title_count : 存储每个节点在不同网络上的度
node_title_outdegree_2015 ： 存储节点在不同网络上的出度
node_title_indegree_2015 ： 存储节点在不同网络上的入度
"""
if __name__ == '__main__':
    print()

    #############################################################################################
    # 以下为构建link prediction数据的部分
    # base = '/Users/laughing/Desktop/newdata/'
    # path = base + 'careerChain_2015.txt'
    # out_path = base + 'format_chain_2015.txt'

    # 第一步：格式化原始链，去重复，筛选包含目标公司和岗位的数据
    # format_data(path, out_path)

    # 第二步：得到不同岗位比较纯粹的职业链
    # get_relax_pure_link('format_chain_2015.txt', 'relax_2015.txt')
    # count_for_pure(base+'relax_2015.txt')

    # 第三步：根据链得到转移矩阵
    # matrix = get_direct_matrix(base+'relax_2015.txt')
    # f = open(base+'matrix_all_2015', 'w')
    # for k in matrix:
    #     for hop in matrix[k]:
    #         f.write('%s %s %d\n'%(k,hop,matrix[k][hop]))
    # f.close()

    # 第四步：筛选节点，首先是得到每个节点在每个网络上的度，出度，入度
    # titles = ['5', '18', '25', '7']
    # filterCommonNodes(base+'matrix_all_2015',titles)

    # 第五步：根据原始图，筛选节点，获得最终的转移矩阵
    # titles = ['5', '18', '25', '7']
    # node_title = read_node_title_count(base+'node_title_count_2015')
    # node_title_outdegree = read_node_title_count(base+'node_title_outdegree_2015')
    # node_title_indegree = read_node_title_count(base+'node_title_indegree_2015')
    # final_nodes = set()
    # for node in node_title:
    #     flag = True
    #     score = 0
    #     if len(node_title[node]) != len(titles) or len(node_title_indegree[node]) != len(titles) or len(node_title_outdegree[node]) != len(titles):
    #         continue
    #     for k in node_title[node]:
    #         score += node_title[node][k]
    #         if node_title[node][k] < 3:
    #             flag = False
    #     if flag and score >= 30:
    #         final_nodes.add(node)
    # print (len(final_nodes))
    # construct_matrix_base_nodes_titles(base+'matrix_all_2015', titles, final_nodes, base+'matrix_2015')
    #############################################################################################

    # 以下为构建边权重预测数据的部分
    base = '/Users/laughing/Desktop/newdata/'
    # 第一步：构建不同时间区间的转移矩阵
    # matrix = get_matrix_between_duration(base+'relax_2015.txt', 2015, 2016)
    # f = open(base+'matrix_15_16', 'w')
    # for k in matrix:
    #     for hop in matrix[k]:
    #         f.write('%s %s %d\n'%(k,hop,matrix[k][hop]))
    # f.close()
    #
    # matrix = get_matrix_between_duration(base + 'relax_2015.txt', 2017, 2020)
    # f = open(base + 'matrix_17', 'w')
    # for k in matrix:
    #     for hop in matrix[k]:
    #         f.write('%s %s %d\n' % (k, hop, matrix[k][hop]))
    # f.close()

    # 第二步：筛选节点
    # titles = ['5', '18', '25', '7']
    # filterCommonNodes(base+'matrix_15_16', titles, suffix='15_16')
    # filterCommonNodes(base+'matrix_17', titles, suffix='17')

    # titles = ['5', '18', '25', '7']
    # node_title1 = read_node_title_count(base+'node_title_count_'+'15_16')
    # node_title_outdegree1 = read_node_title_count(base+'node_title_outdegree_15_16')
    # node_title_indegree1 = read_node_title_count(base+'node_title_indegree_15_16')
    #
    # node_title2 = read_node_title_count(base+'node_title_count_'+'17')
    # node_title_outdegree2 = read_node_title_count(base + 'node_title_outdegree_17')
    # node_title_indegree2 = read_node_title_count(base + 'node_title_indegree_17')
    #
    # nodes = set()
    # for node in node_title1:
    #     flag = True
    #     score = 0
    #     if len(node_title1[node]) != len(titles) or len(node_title_indegree1[node]) != len(titles) or len(node_title_outdegree1[node]) != len(titles):
    #         continue
    #     for k in node_title1[node]:
    #         score += node_title1[node][k]
    #         if node_title1[node][k] < 3:
    #             flag = False
    #             break
    #     if flag and score > 30:
    #         nodes.add(node)
    # print(len(nodes))
    # final_nodes = set()
    # for node in nodes:
    #     flag = True
    #     score = 0
    #     if len(node_title2[node]) != len(titles) or len(node_title_indegree2[node]) != len(titles) or len(node_title_outdegree2[node]) != len(titles):
    #         continue
    #     for k in node_title2[node]:
    #         score += node_title2[node][k]
    #         if node_title2[node][k] < 3:
    #             flag = False
    #             break
    #     if flag and score > 15:
    #         final_nodes.add(node)
    # print(len(final_nodes))
    #
    # f = open('../data/Link_rank_15_17/common_nodes', 'w')
    # for node in final_nodes:
    #     f.write('%s\n'%node)
    # f.close()

    # 第三步：根据选定的节点以及title筛选最后的矩阵
    # final_nodes = set()
    # for line in open('../data/Link_rank_15_17/common_nodes'):
    #     final_nodes.add(line.strip())
    # construct_matrix_base_nodes_titles(base+'matrix_15_16', titles, final_nodes, base+'link_rank_15_16')
    # construct_matrix_base_nodes_titles(base+'matrix_17', titles, final_nodes, base+'link_rank_17')

    common_nodes = set()
    for line in open('../data/All_matrix_2015/all_all'):
        arr = line.strip().split(' ')
        common_nodes.add(arr[1])
        common_nodes.add(arr[2])
    print(len(common_nodes))

    titles = ['5', '18', '25', '7']
    construct_matrix_base_nodes_titles(base + 'matrix_15_16', titles, common_nodes, '../data/Link_rank_15_17/' + 'link_rank_15_16')
    construct_matrix_base_nodes_titles(base + 'matrix_17', titles, common_nodes, '../data/Link_rank_15_17/' + 'link_rank_17')












