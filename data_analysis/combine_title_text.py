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


# 0 text
# 1 title
# 2 title+text
def combine_strategy(title, text):
    substr, length = find_lcsubstr(title, text)
    if length >= len(title):
        return 0
    if length >= len(text):
        return 1
    return 2


# print("数据加载")
data_path = 'filter_res/'
res_path = 'filter_res_combine/'
files = ['clean_stop_train', 'clean_stop_test',
         'clean_train', 'clean_test',
         'half_clean_train', 'half_clean_test',
         'nostop_clean_train', 'nostop_clean_test']
for file in files:
    print(file)
    data = pd.read_csv(data_path + file + '.csv', encoding='GB18030')
    rows = data.shape[0]
    for i in range(0, rows):
        if isinstance(data.loc[i, 'title'], str) and isinstance(data.loc[i, 'text'], str):
            res = combine_strategy(data.loc[i, 'title'], data.loc[i, 'text'])
        elif isinstance(data.loc[i, 'title'], str):
            res = 1
        else:
            res = 0
        if res == 1:
            print("1=", i)
            data.loc[i, 'text'] = data.loc[i, 'title']
        elif res == 2:
            data.loc[i, 'text'] = data.loc[i, 'title'] + data.loc[i, 'text']
    data = data.drop('title', axis=1)
    data.to_csv(res_path + file + "_combine.csv", encoding='GB18030', index=None)
