# 最终在main函数中传入一个维度为6的numpy数组，输出预测值

import os

try:
    import numpy as np
except ImportError as e:
    os.system("sudo pip3 install numpy")
    import numpy as np


def ridge(data):
    x, y = read_data()
    # L  = ||y1 - y||^2 + lamda * ||w||^2
    # w=(X^T  X + λE) ^ −1  X^T y
    # 初始化
    lamda = 1
    w = np.matmul(np.linalg.inv(np.matmul(x.transpose(), x) + lamda * np.ones(6)), np.matmul(x.transpose(), y))
    return w @ data


def piandao(w):
    partial_l1 = [elem for elem in w]
    for i in range(w.shape[0]):
        if partial_l1[i] > 0:
            partial_l1[i] = 1
        elif partial_l1[i] < 0:
            partial_l1[i] = -1
        else:
            partial_l1[i] = 0
    partial_l1 = np.array([partial_l1]).transpose()
    # partial_l1 = partial_l1.transpose()
    return partial_l1


def lasso(data):
    x, y = read_data()
    # 对x每一行进行z-score归一化
    for i in range(x.shape[0]):
        miu = np.mean(x[i])
        sigma = np.std(x[i])
        x[i] = (x[i] - miu) / sigma

    y = np.array([y]).transpose()
    # x : 404*6
    # y : 404*1
    # 初始化
    lamda = 1
    step = 0.01
    epoches = 1000
    num = x.shape[0]
    xlen = x.shape[1]
    w, b = np.zeros((xlen, 1)), 0
    for i in range(epoches):
        # 预测值y_hat
        y_hat = np.dot(x, w) + b  # 404*1

        # 计算dw和db，6*1维和6*1维直接相加
        dw = (np.dot(x.transpose(), (y_hat - y)) / num) + lamda * piandao(w)
        db = np.sum(y_hat - y) / num

        # 更新w，b
        w -= step * dw
        b -= step * db

    # 记得对预测用的数据也要归一化，不然功亏一篑
    data = (data - miu) / sigma
    return np.dot(w.transpose(), data) + b


def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y

