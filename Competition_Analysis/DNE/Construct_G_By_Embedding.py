try:
    from DNE.embedding import *
except ModuleNotFoundError:
    from embedding import *
import pickle


# 从一个文件中获取embedding
def get_embedding_from_path(f_path, dim, word2id):
    emb = Embedding(f_path, dim, word2id)
    model = dict()
    for word in word2id:
        model[word] = emb[str(word)]
    return model


# 从训练文件中获取词表
def get_word2id_from_file(f_path):
    nodes = set()
    titles = set()
    for line in open(f_path):
        arr = line.strip().split(' ')
        if len(arr) == 4:
            nodes.add(int(arr[1]))
            nodes.add(int(arr[2]))
            titles.add(int(arr[0]))
        else:
            nodes.add(int(arr[0]))
            nodes.add(int(arr[1]))
    node2id = dict(zip(sorted(map(int, nodes)), range(len(nodes))))
    title2id = dict(zip(sorted(map(int, titles)), range(len(titles))))
    if len(titles) == 0:
        title2id = None
    return node2id, title2id


def construct_G_for_MDNE(folder, file_name, dim = 128, m_name = 'multi', size = 105224, target_title = 25):
    train_path = '../data/'+ folder + file_name
    node2id, title2id = get_word2id_from_file(train_path)

    base = 'emb/' + folder + m_name + '_' + file_name + '.emb'

    s_path = base + '_s'
    t_path = base + '_t'
    l_path = base + '_l'

    s_model = get_embedding_from_path(s_path, dim, node2id)
    t_model = get_embedding_from_path(t_path, dim, node2id)
    l_model = get_embedding_from_path(l_path, dim, title2id)

    base = 'emb/' + folder + m_name + '_' + file_name + '_T.emb'
    T_s_path = base + '_s'
    T_t_path = base + '_t'
    T_l_path = base + '_l'

    T_s_model = get_embedding_from_path(T_s_path, dim, node2id)
    T_t_model = get_embedding_from_path(T_t_path, dim, node2id)
    T_l_model = get_embedding_from_path(T_l_path, dim, title2id)

    final_S = dict()
    final_T = dict()
    final_L = dict()
    for node in s_model:
        final_S[node] = np.append(s_model[node], T_t_model[node])
        final_T[node] = np.append(t_model[node], T_s_model[node])

    for title in l_model:
        final_L[title] = np.append(l_model[title], T_l_model[title])

    list_s = list()
    list_t = list()
    list_l = list()
    nodes = node2id.keys()
    for node in nodes:
        list_s.append(final_S[node])
        list_t.append(final_T[node])
    for title in title2id.keys():
        list_l.append(final_L[title])

    a = np.asarray(list_s)
    b = np.asarray(list_t)
    c = np.asarray(list_l)

    id = title2id[target_title]
    d = np.multiply(c[id], b)
    res = np.matmul(a, d.T)

    print('a.shape:', a.shape, '. b.shape:', b.shape, '. c.shape', c.shape, '. d.shape', d.shape)

    final = dict()
    t0 = time.time()
    for i in range(len(nodes)):
        count = 0
        for j in range(len(nodes)):
            if res[i][j] < 20 or i == j:
                continue
            else:
                count += 1
                k = str(i) + ' ' + str(j)
                final[k] = res[i][j]
        print(i, count)
    print('size of final = ', len(final))
    t1 = time.time()
    print('time = ', t1 - t0)
    sorted_dict = sorted(final.items(), key=lambda item: item[1], reverse=True)
    t2 = time.time()
    print('time = ', t2 - t1)
    f = open('G/mdne_' + str(target_title) + '.G', 'w')
    for l in sorted_dict[0:size]:
        f.write(l[0] + ' ' + str(l[1]) + '\n')
    f.close()


def construct_G_for_DNE(folder, file_name, dim = 128, m_name = 'asymmetric', size = 105224):
    train_path = '../data/' + folder + file_name
    node2id, title2id = get_word2id_from_file(train_path)
    base = 'emb/' + folder + m_name + '_' + file_name + '.emb'
    s_path = base + '_s'
    t_path = base + '_t'

    s_model = get_embedding_from_path(s_path, dim, node2id)
    t_model = get_embedding_from_path(t_path, dim, node2id)

    base = 'emb/' + folder + m_name + '_' + file_name + '_T.emb'
    T_s_path = base + '_s'
    T_t_path = base + '_t'

    T_s_model = get_embedding_from_path(T_s_path, dim, node2id)
    T_t_model = get_embedding_from_path(T_t_path, dim, node2id)

    final_S = dict()
    final_T = dict()
    for node in s_model:
        final_S[node] = np.append(s_model[node], T_t_model[node])
        final_T[node] = np.append(t_model[node], T_s_model[node])

    nodes = node2id.keys()
    nodes = list(nodes)

    list_s = list()
    list_t = list()
    for node in nodes:
        list_s.append(final_S[node])
        list_t.append(final_T[node])

    a = np.asarray(list_s)
    b = np.asarray(list_t)

    print('a.shape:', a.shape, '. b.shape:', b.shape)
    res = np.matmul(a, b.T)

    final = dict()
    t0 = time.time()
    for i in range(len(nodes)):
        print(i)
        for j in range(len(nodes)):
            if res[i][j] < 20 or i == j:
                continue
            else:
                k = str(i) + ' ' + str(j)
                final[k] = res[i][j]
    print('size of final = ', len(final))
    t1 = time.time()
    print('time = ', t1 - t0)
    sorted_dict = sorted(final.items(), key=lambda item: item[1], reverse=True)
    t2 = time.time()
    print('time = ', t2 - t1)
    title = file_name.split('_')[0]
    f = open('G/dne_'+title+'.G', 'w')
    for l in sorted_dict[0:size]:
        f.write(l[0] + ' ' + str(l[1]) + '\n')
    f.close()


# 这里是为了Verse(a)
def construct_G_for_Verse_a(folder, file_name, dim = 128, m_name = 'asymmetric', size = 105224):
    train_path = '../data/' + folder + file_name
    node2id, title2id = get_word2id_from_file(train_path)
    base = 'emb/' + folder + m_name + '_' + file_name + '.emb'
    s_path = base + '_s'
    t_path = base + '_t'

    s_model = get_embedding_from_path(s_path, dim, node2id)
    t_model = get_embedding_from_path(t_path, dim, node2id)

    final_S = dict()
    final_T = dict()
    for node in s_model:
        final_S[node] = s_model[node]
        final_T[node] = t_model[node]

    nodes = node2id.keys()
    nodes = list(nodes)

    list_s = list()
    list_t = list()
    for node in nodes:
        list_s.append(final_S[node])
        list_t.append(final_T[node])

    a = np.asarray(list_s)
    b = np.asarray(list_t)

    print('a.shape:', a.shape, '. b.shape:', b.shape)
    res = np.matmul(a, b.T)

    final = dict()
    t0 = time.time()
    for i in range(len(nodes)):
        count = 0
        for j in range(len(nodes)):
            if res[i][j] < 10 or i == j:
                continue
            else:
                count += 1
                k = str(i) + ' ' + str(j)
                final[k] = res[i][j]
        print (i, count)
    print('size of final = ', len(final))
    t1 = time.time()
    print('time = ', t1 - t0)
    sorted_dict = sorted(final.items(), key=lambda item: item[1], reverse=True)
    t2 = time.time()
    print('time = ', t2 - t1)
    # f = open('G/VERSE_a_25.G', 'w')
    title = file_name.split('_')[0]
    f = open('G/app_' + title + '.G', 'w')
    for l in sorted_dict[0:size]:
        f.write(l[0] + ' ' + str(l[1]) + '\n')
    f.close()


def construct_G_for_Verse(folder, file_name, dim = 128, m_name = 'verse', size = 105224):
    train_path = '../data/' + folder + file_name
    node2id, title2id = get_word2id_from_file(train_path)
    base = 'emb/' + folder + m_name + '_' + file_name + '.emb'
    s_path = base
    t_path = base

    s_model = get_embedding_from_path(s_path, dim, node2id)
    t_model = get_embedding_from_path(t_path, dim, node2id)

    final_S = dict()
    final_T = dict()
    for node in s_model:
        final_S[node] = s_model[node]
        final_T[node] = t_model[node]

    nodes = node2id.keys()
    nodes = list(nodes)

    list_s = list()
    list_t = list()
    for node in nodes:
        list_s.append(final_S[node])
        list_t.append(final_T[node])

    a = np.asarray(list_s)
    b = np.asarray(list_t)

    print('a.shape:', a.shape, '. b.shape:', b.shape)
    res = np.matmul(a, b.T)

    final = dict()
    t0 = time.time()
    for i in range(len(nodes)):
        print(i)
        for j in range(len(nodes)):
            if res[i][j] < 10 or i == j:
                continue
            else:
                k = str(i) + ' ' + str(j)
                final[k] = res[i][j]
    print('size of final = ', len(final))
    t1 = time.time()
    print('time = ', t1 - t0)
    sorted_dict = sorted(final.items(), key=lambda item: item[1], reverse=True)
    t2 = time.time()
    print('time = ', t2 - t1)
    # f = open('G/VERSE_25.G', 'w')
    title = file_name.split('_')[0]
    f = open('G/verse_' + title + '.G', 'w')
    for l in sorted_dict[0:size]:
        f.write(l[0] + ' ' + str(l[1]) + '\n')
    f.close()


def construct_G_for_Deepwalk(emb_path, dim = 128, size = 105224):
    fr = open(emb_path, 'rb')
    model = pickle.load(fr)

    list_s = list()
    list_t = list()
    for node in model:
        list_s.append(model[node])
        list_t.append(model[node])
    a = np.asarray(list_s)
    b = np.asarray(list_t)

    print('a.shape:', a.shape, '. b.shape:', b.shape)
    res = np.matmul(a, b.T)

    final = dict()
    t0 = time.time()
    for i in range(a.shape[0]):
        count = 0
        for j in range(a.shape[0]):
            if res[i][j] < 3.7 or i == j:
                continue
            else:
                count += 1
                k = str(i) + ' ' + str(j)
                final[k] = res[i][j]
        print(i, count)
    print('size of final = ', len(final))
    t1 = time.time()
    print('time = ', t1 - t0)
    sorted_dict = sorted(final.items(), key=lambda item: item[1], reverse=True)
    t2 = time.time()
    print('time = ', t2 - t1)
    f = open('G/deepwalk_25.G', 'w')
    for l in sorted_dict[0:size]:
        f.write(l[0] + ' ' + str(l[1]) + '\n')
    f.close()


if __name__=='__main__':
    folder = 'All_matrix_2015/'
    title = '25'
    file_name = title + '_all'
    # file_name = '25_all'
    dic = {'5':310613, '25':281860, '7':311567, '18':263725}
    size = dic[title]


    # dim = 64
    # m_name = 'asymmetric'
    # construct_G_for_Verse_a(folder, file_name, dim, m_name, size)

    # dim = 64
    # m_name = 'asymmetric'
    # construct_G_for_DNE(folder, file_name, dim, m_name, size)


    # dim = 128
    # m_name = 'verse'
    # construct_G_for_Verse(folder, file_name, dim, m_name, size)

    construct_G_for_Deepwalk('../emb/deepwalk_'+title+'_all.emb', size=size)

    # construct_G_for_MDNE(folder, 'all_all', 64, target_title=int(title), size=size)

