import numpy as np
import scipy.io as io
import os
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from matplotlib.patches import Rectangle
from matplotlib import cm


def objective_function(rho, X, y, lambda_value):
    residual_term = np.linalg.norm(y - np.dot(X, rho)) ** 2
    sparse_term = lambda_value * np.linalg.norm(rho) ** 2
    return residual_term + sparse_term


if __name__ == '__main__':

    # hyperparameters
    lambda_value_CR = 1

    df = pd.read_excel('assessment.xlsx')
    data = df.values
    data = data.T

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

    num_set = len(set_list)
    overlapping_matrix = np.zeros((num_set, num_set))
    final_non_overlapping_sets = []  # 用于保存每个社区去掉跨社区人物后的最终集合

    for i in range(0, num_set):
        non_overlapping_set_i = set_list[i].copy()  # 初始化当前社区的非重叠集合

        for j in range(num_set):
            if j != i:  # 跳过当前社区
                intersection = non_overlapping_set_i & set_list[j]
                non_overlapping_set_i -= intersection

                if len(intersection) != 0:
                    print(intersection)
                    overlapping_matrix[i, j] = len(intersection)
                    overlapping_matrix[j, i] = len(intersection)
                else:
                    print("无")
        print("社区" + str(i + 1) + "去掉跨社区人物后的最终集合为:")
        print(non_overlapping_set_i)
        final_non_overlapping_sets.append(non_overlapping_set_i)

    for i, final_non_overlapping_set in enumerate(final_non_overlapping_sets):
        print(f"社区{i + 1}: {','.join(map(str, final_non_overlapping_set))}")

    # 找到每两个集合之间的交集
    num_set = len(set_list)
    overlapping_matrix = np.zeros((num_set, num_set))
    non_overlapping_set_list = []
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


subgroup_info = np.zeros((num_set, dim))
for i, set_i in enumerate(set_list):
    for j, item in enumerate(set_i):
        subgroup_info[i, :] += data[:, item - 1]
    subgroup_info[i, :] /= len(set_i)

print("子群信息：\n", subgroup_info)

non_overlapping_info=np.zeros((num_set, dim))
for i, set_i in enumerate(final_non_overlapping_sets):
    for j,item in enumerate(set_i):
        non_overlapping_info[i, :] += data[:, item - 1]
    non_overlapping_info[i, :]/=len(set_i)
print("去掉非重叠后的子群信息：\n", non_overlapping_info)
#df_non_overlapping_info = pd.DataFrame(non_overlapping_info)
#df_non_overlapping_info.to_excel('non_overlapping_info.xlsx',index=False)

consensus = np.ones((num_set, num_set, dim))

for i in range(num_set):
    for j in range(i + 1, num_set):
        consensus[i, j, :] = np.abs(subgroup_info[i] - subgroup_info[j])
        consensus[j, i, :] = np.abs(subgroup_info[i] - subgroup_info[j])

subgroup_consensus = np.zeros(num_set)
for i in range(num_set):
    subgroup_consensus[i] = np.mean(1 - np.mean(consensus[i, :, :], axis=1)) * num_set / (num_set - 1)

print("子群共识度：\n", subgroup_consensus)

group_consensus = 0
total_weight = sum(len(set_i) / num_person for set_i in set_list)
for i, set_i in enumerate(set_list):
    group_consensus += (len(set_i) / num_person) / total_weight * subgroup_consensus[i]  # 权重要归一化

print(len(set_i))
print("群体共识度：\n", group_consensus)

# 定义 x_12 和 x_2 的范围和步长
x_12_values = np.linspace(0, 1, 200)
x_2_values = np.linspace(0, 1, 200)

# 初始化一个二维数组用于存储每个点的 consensus_26 值
consensus_values = np.zeros((len(x_12_values), len(x_2_values)))

# 计算每个点的 consensus_26 值
for i, x_12 in enumerate(x_12_values):
    for j, x_2 in enumerate(x_2_values):
        d_12 = (1 - x_12) * data.T[11, :] + x_12 * non_overlapping_info[5, :]
        SG_2 = (1 - x_2) * non_overlapping_info[1, :] + x_2 * subgroup_info[4, :]
        OC_2 = SG_2 * 4 / 5 + data.T[13, :] / 5
        OC_6 = non_overlapping_info[5, :] * 2 / 3 + d_12 / 3
        consensus_26 = 1 - np.sum(np.abs(OC_2 - OC_6)) / 18
        consensus_values[i, j] = consensus_26

# 绘制图形
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(x_12_values, x_2_values)
surf = ax.plot_surface(X, Y, consensus_values.T, cmap='jet')

# 设置坐标轴标签为latex格式
ax.set_xlabel(r'$\delta_{12}$', fontsize=14)
ax.set_ylabel(r'$\delta_2$', fontsize=14)
ax.set_zlabel(r'$CL_{OC^{26}}$', fontsize=14)

# 调整图形布局，使颜色条位于右侧
plt.subplots_adjust(right=0.8)

# 添加颜色条
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(surf, cax=cbar_ax)

plt.show()

