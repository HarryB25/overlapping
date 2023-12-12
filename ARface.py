import numpy as np
import scipy.io as io
from scipy.optimize import minimize
import matplotlib.pyplot as plt


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

    dim, num_person = data.shape
    result_matrix_CR = np.zeros((num_person, num_person))

    for i in range(0, num_person):
        y = data[:, i]
        D = np.delete(data, i, axis=1)
        rho = minimize(objective_function, np.zeros(num_person - 1), args=(D, y, lambda_value_CR)).x
        rho = np.insert(rho, i, 0)  # 一个测试样本与其他训练样本的关系值放在一列
        result_matrix_CR[:, i] = rho

    np.set_printoptions(linewidth=1000)

    result_matrix_CR1 = np.where(result_matrix_CR > np.mean(result_matrix_CR), result_matrix_CR, 0)
    print(result_matrix_CR1)

    # 获取矩阵的维度
    x_size, y_size = result_matrix_CR1.shape

    # 创建3D图表
    fig = plt.figure(figsize = (10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 生成X和Y的坐标
    x, y = np.meshgrid(np.arange(x_size), np.arange(y_size))

    # 将矩阵展平为1D数组
    z = result_matrix_CR1.flatten()

    # 设置柱子之间的间隔
    dx = 0.8
    dy = 0.8

    # 绘制3D直方图
    ax.bar3d(x.ravel(), y.ravel(), np.zeros_like(z), dx, dy, z)

    # 设置坐标轴标签
    ax.set_xlabel('person id')
    ax.set_ylabel('person id')
    ax.set_zlabel('CR')

    # 显示图表
    plt.show()