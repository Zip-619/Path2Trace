def get_all_data(min_time=2014, max_time=2017):
    path = '/Users/laughing/Desktop/competition_data/TF_data_after_2014.txt'
    all_data = dict()
    target_title = ['5', '7', '18', '25']
    target_com = set()
    for line in open('target_company'):
        target_com.add(line.strip())
    print(len(target_com))
    for line in open(path):
        arr = line.strip().split(' ')
        title = arr[0]
        if title not in target_title:
            continue
        f_node = arr[1]
        t_node = arr[2]
        if f_node not in target_com or t_node not in target_com:
            continue
        time = int(arr[3])
        weight = int(arr[-1])
        if time < min_time or time > max_time:
            continue
        else:
            if title not in all_data:
                all_data[title] = dict()
            key = f_node + ' ' + t_node
            if key not in all_data[title]:
                all_data[title][key] = 0
            all_data[title][key] += weight

    base = '../data/All_matrix_' + str(min_time)
    f = open(base+'/all_all', 'w')
    f_T = open(base+'/all_all_T', 'w')
    for edge_type in all_data:
        for key in all_data[edge_type]:
            f_node = key.split(' ')[0]
            t_node = key.split(' ')[1]
            weight = str(all_data[edge_type][key])
            f.write(edge_type+' '+f_node+' '+t_node+' '+weight+'\n')
            f_T.write(edge_type+' '+t_node+' '+f_node+' '+weight+'\n')
    f.close()
    f_T.close()

    for edge_type in all_data:
        tmp_f = open(base + '/' + edge_type + '_all', 'w')
        tmp_f_T = open(base + '/' + edge_type + '_all_T', 'w')
        for key in all_data[edge_type]:
            f_node = key.split(' ')[0]
            t_node = key.split(' ')[1]
            weight = str(all_data[edge_type][key])
            tmp_f.write(f_node+' '+t_node+' '+weight+'\n')
            tmp_f_T.write(t_node+' '+f_node+' '+weight+'\n')
        tmp_f.close()
        tmp_f_T.close()


def get_Link_Rank_train_data(min_time=2015, max_time=2016):
   ###############################
   # 需要重写
   ###############################
    target_title = ['5', '7', '18', '25']
    # base = '../data/Link_Rank_14_16_17'
    base = '../data/Link_Rank_15_16_17'
    path = '/Users/laughing/Desktop/competition_data/TF_data_after_2014_more_250.txt'
    f = open(base+'/all', 'w')
    f_T = open(base+'/all_T', 'w')
    target_com = set()
    for line in open('target_company'):
       target_com.add(line.strip())
    print(len(target_com))
    for line in open(path):
        arr = line.strip().split(' ')
        f_com = arr[1]
        t_com = arr[2]
        if f_com not in target_com or t_com not in target_com:
            continue
        time = int(arr[3])
        if time < min_time or time > max_time:
            continue
        count = arr[-1]
        if arr[0] in target_title:
            f.write(arr[0] + ' ' + f_com + ' ' + t_com + ' ' + count + '\n')
            f_T.write(arr[0] + ' ' + t_com + ' ' + f_com + ' ' + count + '\n')
    f.close()
    f_T.close()
    for title in target_title:
        tmp_f = open(base + '/' + title + '_'+str(min_time)+'_'+str(max_time), 'w')
        tmp_f_T = open(base + '/' + title + '_'+str(min_time)+'_'+str(max_time)+'_T', 'w')
        for line in open(path):
            arr = line.strip().split(' ')
            f_com = arr[1]
            t_com = arr[2]
            time = int(arr[3])
            if time < min_time or time > max_time:
                continue
            count = arr[-1]
            if arr[0] == title:
                tmp_f.write(f_com + ' ' + t_com + ' ' + count + '\n')
                tmp_f_T.write(t_com + ' ' + f_com + ' ' + count + '\n')
        tmp_f.close()
        tmp_f_T.close()


def get_Link_Rank_test_data(min_time=2017, max_time=2017):
    ###############################
    # 需要重写
    ###############################
    target_title = ['5', '7', '18', '25']
    base = '../data/Link_Rank_15_16_17'
    path = '/Users/laughing/Desktop/competition_data/TF_data_after_2014_more_250.txt'
    for title in target_title:
        tmp_f = open(base + '/' + title + '_'+str(min_time), 'w')
        for line in open(path):
            arr = line.strip().split(' ')
            f_com = arr[1]
            t_com = arr[2]
            time = int(arr[3])
            if time < min_time or time > max_time:
                continue
            count = arr[-1]
            if arr[0] == title:
                tmp_f.write(f_com + ' ' + t_com + ' ' + count + '\n')
        tmp_f.close()


if __name__=='__main__':
    # path = '/Users/laughing/Desktop/competition_data/TF_data_after_2014.txt'
    # edges = dict()
    # coms = dict()
    # for line in open(path):
    #     arr = line.strip().split(' ')
    #     if arr[0] not in edges:
    #         edges[arr[0]] = 0
    #     edges[arr[0]] += 1
    #     if arr[0] not in coms:
    #         coms[arr[0]] = set()
    #     coms[arr[0]].add(arr[1])
    #     coms[arr[0]].add(arr[2])
    # title_index = dict()
    # for line in open('title_index.txt'):
    #     arr = line.strip().split(';')
    #     title_index[arr[1]] = arr[0]
    # for title in edges:
    #     print (title,'\t', title_index[title],'\t',edges[title], '\t',len(coms[title]))

    get_all_data(2015, 2017)
    # get_Link_Rank_train_data(2015, 2017)

    # target_titles = set(['5', '7', '25', '18'])
    # dic = dict()
    # nodes = set()
    # path = '/Users/laughing/Desktop/competition_data/TF_data_after_2014.txt'
    # for line in open(path):
    #     arr = line.strip().split(' ')
    #     year = int(arr[3])
    #     if year < 2016 or year > 2017:
    #         continue
    #     if arr[0] not in target_titles:
    #         continue
    #     if arr[0] not in dic:
    #         dic[arr[0]] = set()
    #     dic[arr[0]].add(arr[1])
    #     dic[arr[0]].add(arr[2])
    #     nodes.add(arr[1])
    #     nodes.add(arr[2])
    # for title in dic:
    #     print(title, len(dic[title]))
    # res = set()
    # f = open('target_company', 'w')
    # for node in nodes:
    #     if node in dic['5'] and node in dic['7'] and node in dic['18'] and node in dic['25']:
    #         res.add(node)
    #         f.write('%s\n'%node)
    # print(len(res))
    # f.close()





    # get_Link_Rank_test_data()
    # path = '/Users/laughing/Desktop/competition_data/TF_data_after_2014.txt'
    # res = dict()
    # times = dict()
    # target_titles = set(['5','7','25','18'])
    # for line in open(path):
    #     time = int(line.strip().split(' ')[3])
    #     if time < 2014 or time > 2017:
    #         continue
    #     if time not in times:
    #         times[time] = 0
    #     times[time] += 1
    #     title = line.strip().split(' ')[0]
    #     if title not in res:
    #         res[title] = 0
    #     res[title]+=1
    #     if title not in target_titles:
    #         continue
    #
    # for k in res:
    #     print(k, '\t',res[k])
    # print('xxxxxxxxxxxx')
    # for k in times:
    #     print(k,'\t',times[k])