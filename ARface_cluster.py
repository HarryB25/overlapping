import numpy as np
import scipy.io as io
import os
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from matplotlib.patches import Rectangle
from matplotlib import cm
from utils.LFM import LFM
from utils.COPRA import *


def objective_function(rho, X, y, lambda_value):
    residual_term = np.linalg.norm(y - np.dot(X, rho)) ** 2
    sparse_term = lambda_value * np.linalg.norm(rho) ** 2
    return residual_term + sparse_term


if __name__ == '__main__':

    # hyperparameters
    lambda_value_CR = 1

    # 读取ar.mat文件
    ar_face = io.loadmat('ar.mat')
    ar_face = ar_face['data']  # ar_face的形状为（50， 40， 3120），50*40的图像，共3120张，每人26张，共120人
    ar_face = ar_face.reshape(50 * 40, 3120)  # 拉直图像，将每张图像变成一列
    ar_face = ar_face.reshape(2000, 120, 26)  # 将ar_face改为（2000, 120, 26），每一列为一个人的图像
    data = ar_face[:, 0, :]

    # 按列映射到[0,1]区间
    # data = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))
    dim, num_person = data.shape
    result_matrix_CR = np.zeros((num_person, num_person))

    for i in range(0, num_person):
        y = data[:, i]
        D = np.delete(data, i, axis=1)
        rho = minimize(objective_function, np.zeros(num_person - 1), args=(D, y, lambda_value_CR)).x
        rho = np.insert(rho, i, 0)  # 一个测试样本与其他训练样本的关系值放在一列
        result_matrix_CR[:, i] = rho

    np.set_printoptions(linewidth=1000)
    # print(result_matrix_CR)

    result_matrix_CR1 = np.where(result_matrix_CR > np.mean(result_matrix_CR), result_matrix_CR, 0)
    print(result_matrix_CR1)

    # 在result_matrix_CR中找到每个人最大的top-k(k=2)个人
    k = 2
    top_k_indices = np.argsort(-result_matrix_CR1, axis=0)[:k, :] + 1  # 将CR系数从大到小排列
    person_id_vector = np.arange(1, num_person + 1)
    top_k_indices = np.vstack((person_id_vector, top_k_indices))
    top_k_indices = top_k_indices.T
    top_k_indices_list = top_k_indices.tolist()
    print("信任关系：")
    print(top_k_indices_list)

    cur_num = 0
    set_list = []
    while cur_num != num_person:
        new_set = set()
        cur_tuple = top_k_indices_list[0]
        new_set.add(cur_tuple[0])
        new_set.add(cur_tuple[1])
        new_set.add(cur_tuple[2])
        top_k_indices_list.remove(cur_tuple)
        cur_num += 1
        i = 0
        while i != len(top_k_indices_list):
            # 如果top_k_indices_list[i]中有两个元素在new_set中，那么将top_k_indices_list[i]中的元素加入new_set
            if len(set(top_k_indices_list[i]) & new_set) >= 2:
                new_set.add(top_k_indices_list[i][0])
                new_set.add(top_k_indices_list[i][1])
                new_set.add(top_k_indices_list[i][2])
                top_k_indices_list.remove(top_k_indices_list[i])
                i = 0
                cur_num += 1
            else:
                i += 1
        set_list.append(new_set)

    # 遍历每两个集合之间的交集，如果交集的元素个数大于2，那么将两个集合合并，并且从set_list中删除一个集合
    i = 0
    len_set_list = len(set_list)
    while i < len_set_list:
        j = 0
        while j < len_set_list:
            if i == j:
                j += 1
                continue
            if len(set_list[i] & set_list[j]) >= 2:
                set_list[i] = set_list[i] | set_list[j]
                # set_list[j]置为空集
                set_list[j] = set()
                j = 0
            else:
                j += 1
        i += 1

    # 删除空集
    i = 0
    while i < len(set_list):
        if len(set_list[i]) == 0:
            set_list.remove(set_list[i])
        else:
            i += 1

    print("社区：")
    print(set_list)

    # 找到每两个集合之间的交集
    num_set = len(set_list)
    overlapping_matrix = np.zeros((num_set, num_set))
    for i in range(0, num_set):
        for j in range(i + 1, num_set):
            print("社区" + str(i + 1) + "和社区" + str(j + 1) + "的跨社区人物为：")
            intersection = set_list[i] & set_list[j]
            if len(intersection) != 0:
                print(intersection)
                assert len(intersection) == 1, "交集中的元素个数不为1"
                overlapping_matrix[i, j] = list(intersection)[0]
                overlapping_matrix[j, i] = list(intersection)[0]
            else:
                print("无")
                overlapping_matrix[i, j] = 0

    print(overlapping_matrix)

    # 绘图，展示多组社区之间的交集，以及每个社区的跨社区人物

    # 创建一个有向图
    G = nx.DiGraph()

    # 添加节点
    for i, s in enumerate(set_list):
        G.add_node(i, label=str(i + 1))

    # 用overlap_matrix表示边的邻接矩阵
    for i in range(0, num_set):
        for j in range(i + 1, num_set):
            if overlapping_matrix[i, j] != 0:
                G.add_edge(i, j, label=str(int(overlapping_matrix[i, j])))

    # 绘制图形
    plt.figure(figsize=(5, 5))
    pos = nx.spring_layout(G, seed=42)  # 使用Spring布局算法进行布局
    edge_labels = nx.get_edge_attributes(G, 'label')
    node_labels = nx.get_node_attributes(G, 'label')
    nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=1500)
    nx.draw_networkx_edges(G, pos, arrows=True)
    nx.draw_networkx_labels(G, pos, labels=node_labels)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.axis('off')
    plt.show()

    subgroup_info = np.zeros((num_set, dim))
    for i, set_i in enumerate(set_list):
        for j, item in enumerate(set_i):
            subgroup_info[i, :] += data[:, item - 1]
        subgroup_info[i, :] /= len(set_i)

    print("子群信息：\n", subgroup_info)

    consensus = np.ones((num_set, num_set, dim))

    for i in range(num_set):
        for j in range(i+1, num_set):
            consensus[i, j, :] = np.abs(subgroup_info[i] - subgroup_info[j])
            consensus[j, i, :] = np.abs(subgroup_info[i] - subgroup_info[j])

    subgroup_consensus = np.zeros(num_set)
    for i in range(num_set):
        subgroup_consensus[i] = np.mean(1-np.mean(consensus[i, :, :], axis=1))*num_set/(num_set-1)

    print("子群共识度：\n", subgroup_consensus)

    group_consensus = 0
    for i, set_i in enumerate(set_list):
        group_consensus += len(set_i)/num_person * subgroup_consensus[i]

    print("群体共识度：\n", group_consensus)

    G_DM = nx.from_numpy_array(result_matrix_CR1)

    labels = {i: i + 1 for i in range(len(G_DM.nodes))}

    # 将标签设置到图中
    nx.set_node_attributes(G_DM, labels, 'label')

    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(G_DM, seed=42)
    edge_labels = nx.get_edge_attributes(G_DM, 'label')
    node_labels = nx.get_node_attributes(G_DM, 'label')
    nx.draw_networkx_nodes(G_DM, pos, node_color='skyblue', node_size=1500)
    nx.draw_networkx_edges(G_DM, pos, arrows=True)
    nx.draw_networkx_labels(G_DM, pos, labels=node_labels)
    nx.draw_networkx_edge_labels(G_DM, pos, edge_labels=edge_labels)
    plt.axis('off')
    plt.show()

    alpha_value = 1.5
    lfm = LFM(G_DM, alpha_value)

    # 执行社区发现
    communities = lfm.execute()

    # 打印结果
    print("Detected Communities:")

    m = len(communities)
    n = max(len(c) for c in communities)
    fig, axes = plt.subplots(m, n, figsize=(20, 20))

    for i, community_nodes in enumerate(communities):
        print(f"Community {i + 1}: {community_nodes}")
        for j, person_id in enumerate(community_nodes):
            face = data[:, person_id-1].reshape(50, 40)
            axes[i, j].imshow(face, cmap='gray')

    # 关闭所有坐标轴
    for ax in axes.flatten():
        ax.axis('off')

    # 显示图像
    fig.show()

    A = array(nx.adjacency_matrix(G_DM).todense())
    degree_s, neighbours, sums = Degree_Sorting(A, len(G_DM.nodes()), G_DM)
    PR = relevance(G_DM, 8, 0.1)
    coms = getcoms(degree_s, neighbours, sums, A, 9, len(G_DM.nodes()), PR, G_DM)

    coms = list(coms.values())

    m = len(coms)
    n = max(len(c) for c in coms)
    fig, axes = plt.subplots(m, n, figsize=(20, 20))

    for i, community_nodes in enumerate(coms):
        print(f"Community {i + 1}: {community_nodes}")
        for j, person_id in enumerate(community_nodes):
            face = data[:, person_id - 1].reshape(50, 40)
            axes[i, j].imshow(face, cmap='gray')

    # 关闭所有坐标轴
    for ax in axes.flatten():
        ax.axis('off')

    # 显示图像
    fig.show()
