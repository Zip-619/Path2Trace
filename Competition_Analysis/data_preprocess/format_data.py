def hop_count(path):
    """
    统计跳槽次数
    """
    hop = dict()
    for line in open(path):
        length = len(line.strip().split('.'))
        if length not in hop:
            hop[length] = 0
        hop[length] += 1
    for k in hop:
        print(k - 1, '\t', hop[k])


def countByYear(path, year):
    """
    统计某一个时间点后的职业生涯练的数目
    """
    hop = dict()
    count = 0
    for line in open(path):
        exp = line.strip().split('.')
        index = -1
        for e in exp[:-1]:
            index += 1
            time = e.split(',')[2]
            if int(time.split('-')[0].split('/')[0]) >= year:
                # print (exp[index:])
                count += 1
                length = len(exp) - index
                if length not in hop:
                    hop[length] = 0
                hop[length] += 1
                break
    print('all number = ', count)
    for k in hop:
        print(k - 1, '\t', hop[k])


def getChainByYear(path, year, outPath):
    """
    得到某一个时间点以后的职业生涯练数据，可能第一段工作经历已经包含了时间节点
    """
    f = open(outPath, 'w')
    for line in open(path):
        exp = line.strip().split('.')
        index = -1
        for e in exp[:-1]:
            index += 1
            time = e.split(',')[2]
            if int(time.split('-')[0].split('/')[0]) >= year or int(time.split('-')[1].split('/')[0]) >= year:
                length = len(exp) - index
                if length > 0:
                    string = '.'.join(exp[index:])
                    f.write(string + '\n')
                break
    f.close()


title_set = set([str(i) for i in range(1, 27)])


def getHopByTitle(path):
    """
    统计不同title之间的流转情况
    """
    res = dict()
    for line in open(path):
        exp = line.strip().split('.')
        for i in range(0, len(exp) - 1):
            j = i + 1
            pre = exp[i]
            after = exp[j]
            pre_name = exp[i].split(',')[1]
            after_name = exp[j].split(',')[1]
            pre_title = pre.split(',')[0]
            after_title = after.split(',')[0]
            if pre_title in title_set and after_title in title_set and pre_name != after_name:
                # if pre_title in title_set and after_title in title_set and pre_title != after_title:
                key = str(pre_title) + '\t' + str(after_title)
                if key not in res:
                    res[key] = 0
                res[key] += 1
    for k in res:
        print(k, '\t', res[k])


def countChain(path):
    """
    计算一个文档中跳槽数等于0的数目
    """
    allcount = 0
    count = 0
    for line in open(path):
        length = line.strip().split('.')
        if length == 1:
            count += 1
        allcount += 1
    print(allcount, count)


def judgeChain(string, company_indexs, title_indexs):
    """
    判断一条链是否可以保留下来，至少包含两个公司之间的流转，包含一个目标岗位的流转
    """
    exps = string.split('.')
    com_set = set([])
    job_set = set([])
    for exp in exps:
        title = exp.split(',')[0]
        company = exp.split(',')[1]
        if title in title_indexs:
            job_set.add(title)
        if company in company_indexs:
            com_set.add(company)
    if len(com_set) >= 2 and len(title_set) >= 1:
        return True
    else:
        return False


def filterChain(path):
    """
    对链进行过滤,确保每个链中至少有两个筛选下来的公司,另外必须保证至少有一个筛选的岗位
    """
    all_count = 0
    company_index = set([])  # 存储所有公司对应的ID
    title_index = set([])  # 存储所有岗位对应的ID
    for i in range(1, 38824):
        company_index.add(str(i))
    for i in range(1, 27):
        title_index.add(str(i))
    true_count = 0
    for line in open(path):
        all_count += 1
        if judgeChain(line, company_index, title_index):
            true_count += 1
    print('包含两个目标公司以及一个目标岗位的公司个数为:', true_count, '总体链的数目为：', all_count)
    return true_count


def partChain(string, company_index):
    """
    输入一个字符串，只保留首尾是目标公司的链
    """
    exps = string.split('.')
    first_index = -1
    i = -1
    for exp in exps:
        i += 1
        company = exp.split(',')[1]
        if company in company_index and first_index == -1:
            first_index = i
            break
    last_index = -1
    index = len(exps)
    for i in range(len(exps) - 1, -1, -1):
        exp = exps[i]
        index -= 1
        company = exp.split(',')[1]
        if company in company_index and last_index == -1:
            last_index = index
            break
    if last_index > first_index:
        target = exps[first_index:last_index + 1]
        return '.'.join(target)
    return ''


def getChainBetweenTarget(path, out_path):
    """
    得到一条链上的只包含目标公司之间的流转
    """
    count = 0
    f = open(out_path, 'w')
    company_index = set([])  # 存储所有公司对应的ID
    title_index = set([])
    for i in range(1, 38824):
        company_index.add(str(i))
    for i in range(1, 27):
        title_index.add(str(i))
    for line in open(path):
        string = partChain(line, company_index)
        if len(string) > 0 and judgeChain(string, company_index, title_index) == True:
            count += 1
            f.write(string)
    f.close()
    print(count)


def boolPureChain(string):
    """
    判断纯粹的跳转链，如果所有的title都相同则是纯粹
    :param string:
    :return:
    """
    title_index = set([])
    for i in range(1, 27):
        title_index.add(str(i))
    exps = string.split('.')
    job_set = set([])
    for exp in exps[:-1]:
        title = exp.split(',')[0]
        if title in title_index:
            job_set.add(title)
    if len(job_set) == 1:
        return True
    return False


def getPureChain(path, out_path):
    count = 0
    f = open(out_path, 'w')
    for line in open(path):
        if boolPureChain(line.strip()):
            f.write(line.strip() + '\n')
            count += 1
    f.close()
    print(count)


def getNeededChain(path):
    """
    做一个限制，所有跳转次数大于10的直接忽略掉,统计非pure链中，某一个title出现的次数大于流转一半的链数
    """
    count = 0
    res = {}
    for string in open(path):
        exps = string.strip().split('.')
        if len(exps) > 2 and len(exps) < 10:
            if boolPureChain(string) == False:
                title_dict = {}
                for exp in exps:
                    title = exp.split(',')[0]
                    if title in title_set:
                        if title not in title_dict:
                            title_dict[title] = 0
                        title_dict[title] += 1
                for v in title_dict.values():  # 其中某一类title站住所有title的一半以上
                    if v >= round(len(exps) / 2):
                        count += 1
                        break
                # f_title = exps[0].split(',')[0]
                # t_title = exps[-1].split(',')[0]
                # if f_title == t_title:
                #     k = len(exps) - 1
                #     if k not in res:
                #         res[k] = 0
                #     res[k] += 1
                #     count += 1
    print(count)


def getNeededChain2(path):
    """
    做一个限制，所有跳转次数大于10的直接忽略掉,统计非pure链中，某一个title出现的次数大于流转一半的链数, 另外统计存在一个title至少出现两次的情况
    """
    count = 0
    newCount = 0
    res = {}
    for string in open(path):
        flag = True
        exps = string.strip().split('.')
        if len(exps) > 2 and len(exps) < 10:
            if boolPureChain(string) == False:
                title_dict = {}
                for exp in exps:
                    title = exp.split(',')[0]
                    if title in title_set:
                        if title not in title_dict:
                            title_dict[title] = 0
                        title_dict[title] += 1
                for v in title_dict.values():  # 其中某一类title站住所有title的一半以上
                    if v >= round(len(exps) / 2):
                        count += 1
                        flag = False
                        break
                if flag == True:
                    newDic = {}
                    index = -1
                    for exp in exps:
                        index += 1
                        title = exp.split(',')[0]
                        if title not in newDic:
                            newDic[title] = []
                        newDic[title].append(index)
                    for k in newDic:
                        if newDic[k][-1] - newDic[k][0] > 1:
                            newCount += 1
                            break
    print(count)
    print(newCount)


def getLINKChain(path, out_path):
    """
    这步比较重要，结果存储在simple_chain_2010.txt
    获取最纯粹的链,对于不纯粹的链，保留某一个title超过半数的
    直接将链拆开，两个相同的链中间夹着的保留
    """
    f = open(out_path, 'w')
    title_index = set([])
    for i in range(1, 27):
        title_index.add(str(i))
    pure_count = 0
    more_count = 0
    between_count = 0
    for line in open(path):
        string = ''
        exps = line.strip().split('.')
        if boolPureChain(line):  # 这是获取最纯粹的链
            for exp in exps:
                title = exp.split(',')[0]
                if title in title_index:
                    string = title + ';' + '.'.join([exp.split(',')[1] for exp in exps])
                    pure_count += 1
                    break   # 这个地方的break导致很多纯粹的链没有被写入
        else:
            flag = True
            title_dict = {}
            index = -1
            for exp in exps:
                index += 1
                title = exp.split(',')[0]
                if title in title_index:
                    if title not in title_dict:
                        title_dict[title] = []
                    title_dict[title].append(index)
            for k in title_dict:  # 对于不纯粹的链，保留某一个title超过半数的
                if len(title_dict[k]) >= round(len(exps) / 2):
                    string = k + ';' + '.'.join([exp.split(',')[1] for exp in exps])
                    more_count += 1
                    flag = False
                    break
            # if flag == True:
            #     string_list = []  # 直接将链拆开，两个相同的链中间夹着的保留
            #     for k in title_dict:
            #         f_index = title_dict[k][0]
            #         t_index = title_dict[k][-1]
            #         if t_index - f_index >= 1 and t_index < len(exps) - 1:
            #             new_string = k + ';' + '.'.join([exp.split(',')[1] for exp in exps[f_index:t_index+2]])
            #             string_list.append(new_string)
            #         if t_index - f_index > 1 and t_index == len(exps) - 1:
            #             new_string = k + ';' + '.'.join([exp.split(',')[1] for exp in exps[f_index:t_index+1]])
            #             string_list.append(new_string)
            #     string = '\n'.join(string_list)
            #     if len(string) > 0:
            #         between_count += 1
    #     if len(string) > 0:
    #         f.write(string + '\n')
    # f.close()
    print(pure_count)
    print(more_count)
    print(between_count)


def simple_Chain(path, out_path):
    count = 0
    f = open(out_path, 'w')
    company_index = set([])  # 存储所有公司对应的ID
    for i in range(1, 38824):
        company_index.add(str(i))
    pre_string = ''
    for line in open(path):
        id = line.strip().split(';')[0]
        ls = []
        exps = line.strip().split(';')[1].split('.')
        for i in range(0, len(exps)):
            if exps[i] not in company_index:
                ls.append(i)
        for i in ls:
            if i - 1 >= 0 and i + 1 < len(exps) and exps[i - 1] == exps[i + 1]:
                exps[i] = exps[i - 1]
        res = []
        for exp in exps:
            if len(res) > 0 and exp == res[-1]:
                continue
            else:
                res.append(exp)
        if len(res) > 1:
            string = id + ';' + '.'.join(res)
            if string != pre_string:
                f.write(string + '\n')
                pre_string = string
        else:
            count += 1
    f.close()
    print(count)


# 对新式的数据类型进行统计
# title; com1.com2....
def countForNew(path):
    count = 0
    res = {}
    for line in open(path):
        count += 1
        length = len(line.strip().split(';')[1].split('.')) - 1
        if length not in res:
            res[length] = 0
        res[length] += 1
    print(count)
    for k in res:
        print(k, '\t', res[k])


def countLinkForCom(path):
    """
    统计每个公司被多少条链包含
    """
    company_index = set([])  # 存储所有公司对应的ID
    for i in range(1, 38824):
        company_index.add(str(i))
    f = open('com_link_count_2014.txt', 'w')
    res = {}
    for line in open(path):
        exps = set(line.strip().split(';')[1].split('.'))
        for exp in exps:
            if exp not in company_index:
                continue
            if exp not in res:
                res[exp] = 0
            res[exp] += 1
    for k in res:
        print(k, '\t', res[k])
        f.write(str(k) + '\t' + str(res[k]) + '\n')
    f.close()


def getChainOfTargetCom(path, number):
    """
    给定一个目标公司集合，保留剩下公司之间的流转
    """
    f = open(path.split('.')[0] + '_' + str(number) + '.txt', 'w')
    dic = {}
    for line in open('com_link_count_2014.txt'):
        k = line.strip().split('\t')[0]
        v = int(line.strip().split('\t')[1])
        if v >= number:
            dic[k] = v
    company_index = dic.keys()
    hop_dic = {}
    print('目标公司的数目 = ', len(company_index))
    for line in open(path):
        title = line.strip().split(';')[0]
        exps = line.strip().split('.')
        count = 0
        indexs = []
        for i in range(0, len(exps)):
            if exps[i] not in company_index:
                exps[i] = '-'
            else:
                indexs.append(i)
                count += 1
        if count > 1:
            string = title + ';' + '.'.join(exps[indexs[0]:indexs[-1] + 1])
            if indexs[-1] - indexs[0] not in hop_dic:
                hop_dic[(indexs[-1] - indexs[0])] = 0
            hop_dic[(indexs[-1] - indexs[0])] += 1
            f.write(string + '\n')
    f.close()
    print('###########')
    for k in hop_dic:
        if k < 10:
            print(k, '\t', hop_dic[k])


def count4eachTitle(path):
    """
    为每个title计算链数
    """
    dic = {}
    for line in open(path):
        id = line.split(';')[0]
        if id not in dic:
            dic[id] = 0
        dic[id] += 1
    for k in dic:
        print(k, '\t', dic[k])


def getMatrixBasedOnChain(path, out_path):
    """
    根据链得到矩阵
    """
    f = open(out_path, 'w')
    matrix = dict()
    for line in open(path):
        title = line.strip().split(';')[0]
        exps = line.strip().split(';')[1].split('.')
        for i in range(0, len(exps) - 1):
            if exps[i] != '-' and exps[i + 1] != '-':
                k = exps[i] + ' ' + exps[i + 1]
                if k not in matrix:
                    matrix[k] = 0
                matrix[k] += 1
    for k in matrix:
        f.write(k + ' ' + str(matrix[k]) + '\n')
    f.close()


def getMatrixBasedOnChainOfTargetList(target, path, out_path):
    """
    给定一个固定的title，然后生成这些title构成的matrix
    """
    f = open(out_path, 'w')
    matrix = dict()
    for title in target:
        matrix[title] = dict()
    for line in open(path):
        title = line.strip().split(';')[0]
        if title in target:
            exps = line.strip().split(';')[1].split('.')
            for i in range(0, len(exps) - 1):
                if exps[i] != '-' and exps[i + 1] != '-':
                    k = exps[i] + ' ' + exps[i + 1]
                    if k not in matrix[title]:
                        matrix[title][k] = 0
                    matrix[title][k] += 1
    for title in target:
        print(title, len(matrix[title]))
        for k in matrix[title]:
            f.write(title + ' ' + k + ' ' + str(matrix[title][k]) + '\n')
    f.close()


def getMatrixBasedOnChainOfTarget(target, path, out_path):
    """
    根据链得到某一个title的matrix
    """
    f = open(out_path, 'w')
    matrix = dict()
    for line in open(path):
        title = line.strip().split(';')[0]
        if title == target:
            exps = line.strip().split(';')[1].split('.')
            for i in range(0, len(exps) - 1):
                if exps[i] != '-' and exps[i + 1] != '-':
                    k = exps[i] + ' ' + exps[i + 1]
                    if k not in matrix:
                        matrix[k] = 0
                    matrix[k] += 1
    for k in matrix:
        f.write(k + ' ' + str(matrix[k]) + '\n')
    f.close()


def getChainOfTargetJob(target, path, out_path):
    """
    得到特定岗位对应的职业生涯链
    """
    f = open(out_path, 'w')
    for line in open(path):
        title = line.strip().split(';')[0]
        if title == target:
            f.write(line.strip() + '\n')
    f.close()


def getSpecificChain(path, out_path=None):
    """
    这步比较重要
    获取最纯粹的链,对于不纯粹的链，保留某一个title超过半数的
    直接将链拆开，两个相同的链中间夹着的保留
    """
    f = open(out_path, 'w')
    title_index = set([])
    for i in range(1, 27):
        title_index.add(str(i))
    pure_count = 0
    more_count = 0
    between_count = 0
    for line in open(path):
        string = ''
        exps = line.strip().split('.')
        if boolPureChain(line):  # 这是获取最纯粹的链
            for exp in exps:
                title = exp.split(',')[0]
                if title in title_index:
                    string = title + ';' + '.'.join([exp.split(',')[1] for exp in exps])
                    pure_count += 1
                    break
        else:
            flag = True
            title_dict = {}
            index = -1
            for exp in exps:
                index += 1
                title = exp.split(',')[0]
                if title in title_index:
                    if title not in title_dict:
                        title_dict[title] = []
                    title_dict[title].append(index)
            for k in title_dict:  # 对于不纯粹的链，保留某一个title超过半数的
                if len(title_dict[k]) >= round(len(exps) / 2):
                    string = k + ';' + '.'.join([exp.split(',')[1] for exp in exps])
                    more_count += 1
                    flag = False
                    break
            if flag == True:
                string_list = []  # 直接将链拆开，两个相同的链中间夹着的保留
                for k in title_dict:
                    f_index = title_dict[k][0]
                    t_index = title_dict[k][-1]
                    if t_index - f_index >= 1 and t_index < len(exps) - 1:
                        new_string = k + ';' + '.'.join([exp.split(',')[1] for exp in exps[f_index:t_index+2]])
                        string_list.append(new_string)
                    if t_index - f_index > 1 and t_index == len(exps) - 1:
                        new_string = k + ';' + '.'.join([exp.split(',')[1] for exp in exps[f_index:t_index+1]])
                        string_list.append(new_string)
                string = '\n'.join(string_list)
                if len(string) > 0:
                    between_count += len(string_list)
        if len(string) > 0:
            f.write(string + '\n')
    f.close()
    print(pure_count)
    print(more_count)
    print(between_count)


if __name__ == '__main__':


    # ###################################################################################
    # # 以下部分是先获取纯净的职业生涯链，然后基于链构造邻接矩阵
    #
    base = '/Users/laughing/Desktop/competition_data/'
    # ###############################################
    # # Step 1 : 从原始的链中截取某一个时间点以后的链
    path = base + 'simple_careerChain.csv'
    out_path = base + 'careerChain_2015.txt'
    getChainByYear(path, 2015, out_path)
    # ###############################################
    # # Step 2 : 统计某个数据集上的跳槽次数，总链数，以及包含指定岗位在目标公司上流转的个数
    # path = base + 'careerChain_2014.txt'
    # # hop_count(path)
    # filterChain(path)
    # ###############################################
    # # Step 3 : 截断职业生涯链，保证首尾都是目标公司
    # path = base + 'careerChain_2014.txt'
    # out_path = base + 'careerChain_2014_1.txt'
    # getChainBetweenTarget(path, out_path)
    # ###############################################
    # # Step 4 : 获取某个特定岗位一些的链
    # path = base + 'careerChain_2014_1.txt'
    # out_path = base + 'simple_chain_2014.txt'
    # getLINKChain(path, out_path)
    # countForNew(out_path)
    # ###############################################
    # # Step 5 : 将链简化，计算每个公司被包含的链数
    # path = base + 'simple_chain_2014.txt'
    # out_path = base + 'chain_2014.txt'
    # simple_Chain(path, out_path)
    # countLinkForCom(out_path)
    # ###############################################
    # # Step 6 : 得到特定公司之间的链，比如在保留的链中被400条链包含的公司之间的流转
    # path = base + 'chain_2014.txt'
    # getChainOfTargetCom(path, 400)
    # path = base + 'chain_2014_400.txt'
    # path = base + '2010/chain_2010_500.txt'
    # count4eachTitle(path)
    # ###############################################
    # # Step 7 : 进一步筛选
    # ls = set(['5', '7', '25', '18'])
    # path = base + 'chain_2014_400.txt'
    # out_path = base + 'matrix_2014.txt'
    # getMatrixBasedOnChainOfTargetList(ls, path, out_path)
    #
    # ##################################################################################


