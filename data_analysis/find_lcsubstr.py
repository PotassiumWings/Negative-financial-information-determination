import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt

matplotlib.use('TkAgg')
import matplotlib.mlab as mlab
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from lightgbm import plot_importance
import codecs


def find_lcsubstr(s1, s2):
    m = [[0 for i in range(len(s2) + 1)] for j in range(len(s1) + 1)]  # 生成0矩阵，为方便后续计算，比字符串长度多了一列
    mmax = 0  # 最长匹配的长度
    p = 0  # 最长匹配对应在s1中的最后一位
    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i] == s2[j]:
                m[i + 1][j + 1] = m[i][j] + 1
                if m[i + 1][j + 1] > mmax:
                    mmax = m[i + 1][j + 1]
                    p = i + 1
    return s1[p - mmax:p], mmax


# print("数据加载")
data = pd.read_csv('data/train.csv', encoding='GB18030')
data_arr = data.values
rows = data.shape[0]
cnt_1 = 0
cnt_2 = 0
cnt = 0
for i in range(0, rows):
    if isinstance(data_arr[i, 1], str) and isinstance(data_arr[i, 2], str):
        s, l = find_lcsubstr(data_arr[i, 1], data_arr[i, 2])
        if l != 0 and l == len(data_arr[i, 1]):
            cnt_1 += 1
            # print(s + ":: len=" + str(l))
        if l != 0 and l == len(data_arr[i, 2]):
            cnt_2 += 1
        if l != 0 and (l == len(data_arr[i, 1]) or l == len(data_arr[i, 2])):
            cnt += 1

print(" cnt_1 = " + str(cnt_1))
print(" cnt_2 = " + str(cnt_2))
print(" cnt = " + str(cnt))
print(" rows = " + str(rows))
