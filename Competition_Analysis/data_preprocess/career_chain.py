# coding=utf-8

import functools
import re

from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession


class Stack(object):
    # 初始化栈为空列表
    def __init__(self):
        self.items = []

    # 判断栈是否为空，返回布尔值
    def is_empty(self):
        return self.items == []

    # 返回栈顶元素
    def peek(self):
        return self.items[len(self.items) - 1]

    # 返回栈的大小
    def size(self):
        return len(self.items)

    # 把新的元素堆进栈里面（程序员喜欢把这个过程叫做压栈，入栈，进栈……）
    def push(self, item):
        self.items.append(item)

    # 把栈顶元素丢出去（程序员喜欢把这个过程叫做出栈……）
    def pop(self):
        return self.items.pop()

    # 直接返回列表中的所有元素
    def list(self):
        return self.items

    def list_name(self):
        return [l.name for l in self.list()]

    def list_node(self):
        res = []
        for l in self.items:
            string = ','.join([l.title, l.name, l.start_time + '-' + l.end_time])
            res.append(string)
        return res


class Exp_node:

    def __init__(self, id, title, name, start_time, end_time):
        self.id = id
        self.title = title
        self.name = name
        self.start_time = start_time
        self.end_time = end_time
        self.childs = []
        self.father = []

    def to_string(self):
        return self.title + ';' + self.name + ';' + self.start_time + ';' + self.end_time + '; child: ' + '.'.join(
            a.name for a in self.childs) + '; father: ' + '.'.join(a.name for a in self.father)


class Exp_to_Chain:

    def __init__(self, string):
        self.nodes = []
        id = string.split(';')[0]
        exps = string.split(';')[1].split('.')
        # 以下部分是得到所有有意义的节点
        for exp in exps:
            if len(exp.split(',')) > 2:
                title = exp.split(',')[0]
                name = exp.split(',')[1]
                start_time = exp.split(',')[2].split('-')[0]
                end_time = exp.split(',')[2].split('-')[1]
                if len(start_time) > 0 and len(end_time) > 0:
                    node = Exp_node(id, title, name, start_time, end_time)
                    self.nodes.append(node)
        self.sort_nodes()
        self.construct_effect_hop()

    # 比较时间，用于在字典排序时使用
    def compare_time(self, node1, node2):
        t1 = node1.start_time
        t2 = node2.start_time
        y1 = int(t1.split('/')[0])
        m1 = int(t1.split('/')[1])
        y2 = int(t2.split('/')[0])
        m2 = int(t2.split('/')[1])
        if y1 > y2:
            return 1
        elif y1 < y2:
            return -1
        else:
            if m1 > m2:
                return 1
            elif m1 < m2:
                return -1
            else:
                return 0

    # 计算两个时间之间的差值，以月为单位
    def mulTimeByMonth(self, t1, t2):
        big_y = int(t1.split('/')[0])
        big_m = int(t1.split('/')[1])
        small_y = int(t2.split('/')[0])
        small_m = int(t2.split('/')[1])
        return (big_y - small_y) * 12 + (big_m - small_m)

    # 对所有节点根据工作起始时间进行排序
    def sort_nodes(self):
        try:
            ls = sorted(self.nodes, key=functools.cmp_to_key(self.compare_time))  # 针对python3的写法
        except:
            ls = sorted(self.nodes, self.compare_time)  # 针对python2的写法
        self.nodes = ls

    # 构建有效的跳转，给节点标记父节点和子节点
    def construct_effect_hop(self):
        theta_d = 3
        theta_a = -2
        for i in range(0, len(self.nodes) - 1):
            for j in range(i + 1, len(self.nodes)):
                r_ie = self.nodes[i].end_time
                r_js = self.nodes[j].start_time
                delay = self.mulTimeByMonth(r_js, r_ie)
                if delay > theta_a and delay < theta_d:
                    self.nodes[i].childs.append(self.nodes[j])
                    self.nodes[j].father.append(self.nodes[i])
                if delay >= theta_d:
                    break

    # 从一个节点出发找到所有的路径
    def dfs(self, node):
        result = list()
        stack = Stack()
        stack.push(node)
        visited = set([])
        while stack.size() > 0:
            node = stack.peek()
            flag = True
            # print (stack.list_name())
            for left in node.childs:
                if left not in visited:
                    stack.push(left)
                    flag = False
                    break
            if flag == True:
                top_item = stack.peek()
                if len(node.childs) == 0:
                    # print (len(stack.list()))
                    # for l in stack.list():
                    #     print (l.name)
                    result.append(stack.list_node())
                    # result.append(stack.list())
                stack.pop()
                visited.add(top_item)
        return result

    # 从所有的无父节点出发寻找路径
    def getChain(self):
        all_chain = []
        if len(self.nodes) > 0:
            id = self.nodes[0].id
            for node in self.nodes:
                if len(node.father) == 0:
                    res = self.dfs(node)
                    for r in res:
                        if len(r) > 1:  # 至少存在两个节点的路径才有意义
                            all_chain.append(id + ';' + '.'.join(r))
        return all_chain

    def to_string(self):
        for node in self.nodes:
            print(node.to_string())


# connect to spark
def connect_spark():
    conf = SparkConf().setMaster("local[*]")
    conf = conf.setAppName('turn over')
    sc = SparkContext(conf=conf)
    spark = SparkSession \
        .builder \
        .appName("turn over") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()
    return sc, spark


target_names = set([])  # 存储所有出现次数超过500次的公司名称
for line in open('id_more_500'):  # 所有超过500的公司
    target_names.add(line.strip())


# 第一步：筛选简历中包含超过两个目标公司
def judgeContainTwoTargetCom(string):
    exp = string.split(';')[1].split('.')
    target_com = set([])
    for e in exp:
        if len(e.split(',')) > 3:
            name = e.split(',')[1]
            if name in target_names:
                target_com.add(name)
    if len(target_com) > 1:
        return True
    else:
        return False


# 第二步：去掉简历中的工作描述并且将title的名称改掉
def removeWorkDes(string):
    id = string.split(';')[0]
    exp = string.split(';')[1].split('.')
    ls = []
    for e in exp:
        if len(e.split(',')) > 3:
            title = e.split(',')[0]
            title = formatTitle(title)
            name = e.split(',')[1]
            time = e.split(',')[2]
            ls.append(title + ',' + name + ',' + time)
    return id + ';' + '.'.join(ls)


dic_more = {}
for line in open('cb_more'):
    id = re.compile('|'.join(line.strip().split(',')), re.IGNORECASE)
    dic_more[id] = line.split(',')[0]
dic_one = {}
for line in open('cb_one'):
    id = re.compile('|'.join(line.strip().split(',')), re.IGNORECASE)
    dic_one[id] = line.split(',')[0]


# 根据title词表将title名称换掉
def formatTitle(x_title):
    flag = False
    for re_str in dic_more:
        if len(re_str.findall(x_title)) > 0:
            x_title = dic_more[re_str]
            flag = True
            break
    if flag == False:
        for re_str in dic_one:
            # 左右两边都加上空格，
            if len(re_str.findall(' ' + x_title + ' ')) > 0:
                x_title = dic_one[re_str]
                flag = True
                break
    # 如果没有匹配上的，在最前面加上两个*
    if flag == False:
        x_title = '**' + x_title
    return x_title


# 第三步：获取职业生涯链
def getCareerChain(string):
    exp_to_chain = Exp_to_Chain(string.strip())
    result = exp_to_chain.getChain()
    return result


company_id_index = dict()
for line in open('company_index_more_500'):
    name = line.strip().split(';')[0]
    id = line.strip().split(';')[1]
    company_id_index[name] = id

title_id_index = dict()
for line in open('title_index'):
    title = line.strip().split(';')[0]
    id = line.strip().split(';')[1]
    title_id_index[title] = id


def removeUnTarget(string):
    exps = string.strip().split(';')[1].split('.')
    target_com = set([])
    target_title = set([])
    for exp in exps:
        name = exp.split(',')[1]
        title = exp.split(',')[0]
        if name in company_id_index:
            target_com.add(name)
        if title in title_id_index:
            target_title.add(title)
    if len(target_com) > 1 and len(target_title) > 0:
        return True
    else:
        return False


def formatField(string):
    # id = string.split(';')[0]
    exps = string.split(';')[1].split('.')
    ls = []
    for exp in exps:
        arr = exp.split(',')
        title = arr[0]
        name = arr[1]
        time = arr[2]
        if title in title_id_index:
            title = title_id_index[title]
        if name in company_id_index:
            name = company_id_index[name]
        ls.append(','.join([title, name, time]))
    return '.'.join(ls)


if __name__ == '__main__':
    sc, spark = connect_spark()

    # One：去掉经历描述，只选择存在两个目标公司的简历
    # base = 'extHR/zhangle18/link/*'
    # rdd = sc.textFile(base).map(lambda x: x.encode('utf-8')).filter(lambda x: len(x.split(';')) == 2).filter(
    #     len(lambda x: x.split(';')[1].split('.')) > 1).filter(judgeContainTwoTargetCom).map(removeWorkDes).coalesce(
    #     20).saveAsTextFile('extHR/zhangle18/targetLink')

    # Two：抽取职业生涯链
    # base = 'extHR/zhangle18/targetLink/*'
    # rdd = sc.textFile(base).map(lambda x: x.encode('utf-8')).flatMap(getCareerChain).saveAsTextFile('extHR/zhangle18/careerChain')

    # Three : 替换所有的公司名称和岗位名称
    base = 'extHR/zhangle18/careerChain/*'
    rdd = sc.textFile(base).map(lambda x: x.encode('utf-8')).filter(removeUnTarget).map(formatField).saveAsTextFile(
        'extHR/zhangle18/simple_careerChain')
