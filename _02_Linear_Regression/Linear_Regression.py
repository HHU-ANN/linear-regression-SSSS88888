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
    # 对x进行z-score归一化
    miu = np.mean(x)
    sigma = np.std(x)
    for i in range(x.shape[0]):
        x[i] = (x[i] - miu) / sigma

    y = np.array([y]).transpose()
    # y_miu = np.mean(y)
    # y_sigma = np.std(y)
    # y = (y - y_miu) / y_sigma

    # x : 404*6
    # y : 404*1
    # 初始化
    lamda = 1
    step = 0.01
    epochs = 1000
    num = x.shape[0]
    xlen = x.shape[1]
    w, b = np.zeros((xlen, 1)), 0
    for _ in range(epochs):
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
    res = np.dot(w.transpose(), data) + b
    # res = res * y_sigma + y_miu
    return res


def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y

# features = np.array([
#     [2.0133330e+03, 1.6400000e+01, 2.8932480e+02, 5.0000000e+00, 2.4982030e+01, 1.2154348e+02],
#     [2.0126670e+03, 2.3000000e+01, 1.3099450e+02, 6.0000000e+00, 2.4956630e+01, 1.2153765e+02],
#     [2.0131670e+03, 1.9000000e+00, 3.7213860e+02, 7.0000000e+00, 2.4972930e+01, 1.2154026e+02],
#     [2.0130000e+03, 5.2000000e+00, 2.4089930e+03, 0.0000000e+00, 2.4955050e+01, 1.2155964e+02],
#     [2.0134170e+03, 1.8500000e+01, 2.1757440e+03, 3.0000000e+00, 2.4963300e+01, 1.2151243e+02],
#     [2.0130000e+03, 1.3700000e+01, 4.0820150e+03, 0.0000000e+00, 2.4941550e+01, 1.2150381e+02],
#     [2.0126670e+03, 5.6000000e+00, 9.0456060e+01, 9.0000000e+00, 2.4974330e+01, 1.2154310e+02],
#     [2.0132500e+03, 1.8800000e+01, 3.9096960e+02, 7.0000000e+00, 2.4979230e+01, 1.2153986e+02],
#     [2.0130000e+03, 8.1000000e+00, 1.0481010e+02, 5.0000000e+00, 2.4966740e+01, 1.2154067e+02],
#     [2.0135000e+03, 6.5000000e+00, 9.0456060e+01, 9.0000000e+00, 2.4974330e+01, 1.2154310e+02]
#     ])
#
# labels = np.array([41.2, 37.2, 40.5, 22.3, 28.1, 15.4, 50. , 40.6, 52.5, 63.9])
#
#
# max = 0
# best_l = 0
# for lamda in range(1, 100):
#     count = 0
#
#     for i in range(features.shape[0]):
#         if abs(lasso(features[i], lamda) - labels[i]) <= 10: count += 1
#         if count == 10:
#             print("10个满足，lamda={}".format(lamda))
#     if count > max:
#         max = count
#         best_l = lamda
#     print("{}/1000 {} {}".format(lamda, count, best_l))
#
# print(max)
# print(best_l)