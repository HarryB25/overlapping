# coding=utf-8
import random

from numpy import *
import time
import copy
import networkx as nx
import matplotlib.pyplot as plt


# 每个节点的度数，每个节点的邻居节点，总边数
def Degree_Sorting(Adjmartrix, vertices, G):
    degree_s = [[i, 0] for i in range(vertices)]
    neighbours = [[] for i in range(vertices)]
    sums = 0
    for i in range(vertices):
        for j in range(vertices):
            if Adjmartrix[i][j] > 0:
                degree_s[i][1] += 1
                sums += 1
                neighbours[i].append(j)
    return degree_s, neighbours, sums / 2


def Entropy(lanum, num):  # 邻居中各标签的个数和邻居的个数
    sum = 0
    #     print(lanum)
    for i in lanum:
        #         print(i)
        pl = lanum[i] / num
        sum = sum + pl * log10(pl) * len(lanum)
    sum = -sum
    return sum


def Propagate(x, PR, old, new, neighbours, v, asynchronous):
    # new[x] = {}
    # 洗牌保证随机性（都相等的情况）
    random.shuffle(neighbours[x])
    # 依据邻结点标签集更新该节点
    #       label_new = [{} for i in range(vertices)]
    #     label_old = [{i: 1} for i in range(vertices)]

    #     time.sleep(60)
    for eachpoint in neighbours[x]:  # 取出x节点的一个邻居
        #         print(eachpoint,PR)
        for eachlable in old[eachpoint]:  # old[eachpoint]把邻居的字典取出, eachlable字典中的键（标签）
            b = old[eachpoint][eachlable]  # 取出每个标签对应的值，隶属度的值
            if eachlable in new[x]:
                new[x][eachlable] += b * PR[eachpoint]
            else:
                new[x].update({eachlable: b * PR[eachpoint]})
            if asynchronous:
                old[x] = copy.deepcopy(new[x])
    #     print(new[x],"hahahah")
    for i in new[x]:
        new[x][i] /= len(neighbours[x])

    Normalize(new[x])
    #     print(new[x],"hahahah")
    # print new[x]
    maxb = 0.0
    maxc = 0
    t = []
    # 去除小于1/v的候选项，若均小于则''选b最大的赋值''，否则规范化
    for each in new[x]:
        if new[x][each] < 1 / float(v):
            t.append(each)
            if new[x][each] >= maxb:  # 取最后一个
                # if new[x][each] > maxb:#取第一个
                maxb = new[x][each]  # 最大的隶属度
                maxc = each  # 最大隶属度的标签
    for i in range(len(t)):
        del new[x][t[i]]
    if len(new[x]) == 0:
        new[x][maxc] = 1
    else:
        Normalize(new[x])


def Normalize(x):
    sums = 0.0
    for each in x:
        sums += x[each]
    for each in x:
        if sums != 0:
            x[each] = x[each] / sums


def id_l(l):
    ids = []
    for each in l:
        ids.append(id_x(each))  # [[标签1，标签2.。。],[标签1，标签2.。]...]
    return ids


def id_x(x):
    ids = []
    for each in x:
        ids.append(each)
    return ids


def count(l):
    counts = {}
    for eachpoint in l:
        for eachlable in eachpoint:
            if eachlable in counts:
                n = counts[eachlable]
                counts.update({eachlable: n + 1})
            else:
                counts.update({eachlable: 1})
    return counts


def mc(cs1, cs2):
    # print cs1,cs2
    cs = {}
    for each in cs1:
        if each in cs2:
            cs[each] = min(cs1[each], cs2[each])
    return cs


def modularity1(coms, G):
    partition = [list(i) for i in coms.values()]
    print(partition)
    m = len(G.edges(None, False))
    a = []
    e = []
    for community in partition:
        t = 0.0
        for node in community:
            t += len([x for x in G.neighbors(node)])
        a.append(t / (2 * m))
    for community in partition:
        t = 0.0
        for i in range(len(community)):
            for j in range(len(community)):
                if (G.has_edge(community[i], community[j])):
                    t += 1.0
        e.append(t / (2 * m))

    q = 0.0
    for ei, ai in zip(e, a):
        q += (ei - ai ** 2)
    return q


def Modulartiy(A, coms, sums, vertices):
    Q = 0.0
    for eachc in coms:
        li = 0
        for eachp in coms[eachc]:
            for eachq in coms[eachc]:
                li += A[eachp][eachq]
        li /= 2
        di = 0
        for eachp in coms[eachc]:
            for eachq in range(vertices):
                di += A[eachp][eachq]
        Q = Q + (li - (di * di) / (sums * 4))
    Q = Q / float(sums)
    return Q


def ExtendQ(A, coms, sums, k, o):
    # k-每个节点的度数 o-每个节点属于的社区数
    s = float(2 * sums)
    k = sorted(k, key=lambda x: x[0], reverse=False)
    at = 0
    kt = 0
    EQ = 0.0
    for eachc in coms:
        for eachp in coms[eachc]:
            for eachq in coms[eachc]:
                at += A[eachp][eachq] / float(o[eachp] * o[eachq])
                kt += k[eachp][1] * k[eachq][1] / float(o[eachp] * o[eachq])
    EQ = at - kt / s
    return EQ / s


# def InitEgoNet(G):
#     nodes = list(G.nodes())
#     egoNets = [list(G.neighbors(i)) for i in nodes]
#     for i in range(len(egoNets)):
#         egoNets[i].append(nodes[i])
#     return egoNets


# 构建自我网络{节点1:[节点1自我网络的邻居],........}
def InitEgoNet(G):
    nodes = list(G.nodes())
    egoNets = {i: list(G.neighbors(i)) for i in nodes}
    for i in egoNets:
        egoNets[i].append(i)
    return egoNets


# 计算相关性
def relevance(G, maxStep, alpha):
    PR = []
    egoNet = InitEgoNet(G)
    #     print(egoNet)
    V = list(G.nodes())
    for v in V:
        PRv = {x: 0 for x in egoNet[v]}
        PRv[v] = 1.0
        for k in range(maxStep):
            tmp = {x: 0 for x in egoNet[v]}
            for i in egoNet[v]:
                for j in list(G.neighbors(i)):
                    if j not in tmp:
                        tmp[j] = alpha * PRv[i] / len(list(G.neighbors(i)))
                    #                     print(alpha*PRv[i]/len(list(G.neighbors(i))))
                    tmp[j] += alpha * PRv[i] / len(list(G.neighbors(i)))
            tmp[v] += (1 - alpha)
            PRv = tmp
        PR.append(PRv)
    return PR


def getcoms(degree_s, neighbours, sums, A, v, vertices, PR, G):
    label_new = [{} for i in range(vertices)]
    label_old = [{i: 1} for i in range(vertices)]  # 刚开始每个节点的标签是自己，隶属度为1
    minl = {}
    oldmin = {}
    flag = True  # asynchronous异步
    itera = 0  # 迭代次数
    start = time.time()  # 计时
    # 同异步迭代过程
    while True:
        '''
        if flag:
            flag = False
        else:
            flag = True
        '''
        itera += 1

        sortEn = {}
        for x in degree_s:
            x1 = x[0]
            a = [label_old[i] for i in neighbours[x1]]
            a.append(label_old[x1])
            en = Entropy(count(a), len(neighbours[x1]))
            #             sortEn.append({x1:en})
            sortEn.update({x1: en})
        #         print(sortEn)
        sortEnId = sorted(sortEn.items(), key=lambda kv: kv[1])
        #         sortEn = sorted(sortEn, key=lambda x: x.value(), reverse=True)
        #         print(sortEnId)

        #         time.sleep(60)

        # degree_s：[[节点，度数]。。。。。], neighbours = [[第1个节点的所有邻居]，[第2个节点的所有邻居].。。]
        for each in sortEnId:
            #             print(PR[each[0]],"11111111111")
            Propagate(each[0], PR[each[0]], label_old, label_new, neighbours, v, flag)
        if id_l(label_old) == id_l(label_new):
            minl = mc(minl, count(label_new))
        else:
            minl = count(label_new)
        if minl != oldmin:
            label_old = label_new
            oldmin = minl
        else:
            break
    print(itera, label_old)  # 迭代次数
    coms = {}
    sub = {}
    for each in range(vertices):
        ids = id_x(label_old[each])
        for eachc in ids:
            if eachc in coms and eachc in sub:
                coms[eachc].append(each)
                # elif :
                sub.update({eachc: set(sub[eachc]) & set(ids)})
            else:
                coms.update({eachc: [each]})
                sub.update({eachc: ids})
    print('lastiter', coms)
    # 获取每个节点属于的标签数
    o = [0 for i in range(vertices)]
    for eachid in range(vertices):
        for eachl in coms:
            if eachid in coms[eachl]:
                o[eachid] += 1
    # 去重取标签
    for each in sub:
        if len(sub[each]):
            for eachc in sub[each]:
                if eachc != each:
                    coms[eachc] = list(set(coms[eachc]) - set(coms[each]))

    # 改：
    key = [i for i in coms]
    for i in key:
        if len(coms[i]) == 0:
            coms.pop(i)

    # 标签整理
    clusterment = [0 for i in range(vertices)]
    a = 0
    for eachc in coms:
        if len(coms[eachc]) != 0:
            for e in coms[eachc]:
                clusterment[e] = a + 1
            a += 1
    degree_s = sorted(degree_s, key=lambda x: x[0], reverse=False)
    elapsed = (time.time() - start)

    print('t=', elapsed)
    print('result=', coms)
    print('clusterment=', clusterment)
    print('Q =', Modulartiy(A, coms, sums, vertices))
    print('Q2 =', modularity1(coms, G))
    print('EQ =', ExtendQ(A, coms, sums, degree_s, o))
    return coms


if __name__ == '__main__':
    G = nx.read_gml("./football.gml", label="id")

    # 绘制图形
    plt.figure(figsize=(20, 20))
    pos = nx.spring_layout(G, seed=42)  # 使用Spring布局算法进行布局
    edge_labels = nx.get_edge_attributes(G, 'label')
    node_labels = nx.get_node_attributes(G, 'label')
    nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=1500)
    nx.draw_networkx_edges(G, pos, arrows=True)
    nx.draw_networkx_labels(G, pos, labels=node_labels)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.axis('off')
    plt.show()

    A = array(nx.adjacency_matrix(G).todense())
    degree_s, neighbours, sums = Degree_Sorting(A, len(G.nodes()), G)
    PR = relevance(G, 8, 0.85)
    getcoms(degree_s, neighbours, sums, A, 9, len(G.nodes()), PR, G)

