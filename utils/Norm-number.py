from sklearn import preprocessing
import numpy as np
import math
from sklearn.preprocessing import StandardScaler


def map(data,MIN,MAX):
    """
    归一化映射到任意区间
    :param data: 数据
    :param MIN: 目标数据最小值
    :param MAX: 目标数据最小值
    :return:
    """
    d_min = np.min(data)    # 当前数据最大值
    d_max = np.max(data)    # 当前数据最小值
    # y1 = data - d_min
    # y = (MAX-MIN)/(d_max-d_min) * (data - d_min)
    return MIN +(MAX-MIN)/(d_max-d_min) * (data - d_min)

x = np.array([0.988, 0.987, 0.989, 0.984])
# 标准化
# x_scaled = preprocessing.scale(x)
# print(x)
# print(x_scaled)

# z-score
# x = x.reshape(-1, 1)
# scaler = StandardScaler()
# z_score = scaler.fit_transform(x)
# print(x)
# print(z_score)

# softmax
# x_softmax = np.exp(x)/sum(np.exp(x))
# print(x_softmax)

# # 归一化到【0.5,1】
# x_1 = map(x, 0.5, 1)
# print(x_1)

# log(数值，底数)
x1 = math.log(1-0.4587, 0.9)
x2 = math.log(1-0.0706, 0.9)
x3 = math.log(1-0.0396, 0.9)
x4 = math.log(1 - 0.2709, 0.9)
print(x1)
print(x2)
print(x3)
print(x4)
