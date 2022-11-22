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

# print("数据加载")
data = pd.read_csv('data/train.csv', encoding='GB18030')
# print(data.head())
rows = data.shape[0]

def func(words):
    for j in range(0, len(words)):
        for k in range(j+1, len(words)):
            if words[j] in words[k] or words[k] in words[j]:
                return True
    return False

flag_1 = 1
flag_2 = 1

def func_and_in_key(words, key_words):
    global flag_1
    global flag_2
    cnt_j = 0
    cnt_k = 0
    flag = False
    cnt_eq = 0
    for j in range(0, len(words)):
        for k in range(0, len(words)):
            if k != j and words[j] in words[k]:
                for t in range(0, len(key_words)):
                    if words[j] == words[k] and words[j] == key_words[t]:
                        cnt_eq = 1
                        print(words)
                        print(key_words)
                        flag = True
                        break
                    if words[j] == key_words[t]:
                        cnt_j = 1
                        flag = True
                        if flag_1:
                            print(words)
                            print(key_words)
                            flag_1 = 0
                        break
                    if words[k] == key_words[t]:
                        cnt_k = 1
                        flag = True
                        if flag_2:
                            print(words)
                            print(key_words)
                            flag_2 = 0
                        break
    return flag, cnt_j, cnt_k, cnt_eq

cnt_not_null_func = 0
cnt_not_null = 0
cnt_null = 0
cnt_null_0 = 0

cnt_not_null_keynull = 0
cnt_not_null_notin = 0
cnt_not_null_j = 0
cnt_not_null_k = 0
cnt_not_null_eq = 0

for i in range(0, rows):
    if isinstance(data.loc[i, 'entity'], str):
        cnt_not_null += 1
        if func(data.loc[i, 'entity'].split(';')):
            # print(data.loc[i, 'entity'].split(';'))
            cnt_not_null_func += 1
            if isinstance(data.loc[i, 'key_entity'], str):
                flag, c_j, c_k, c_eq = func_and_in_key(data.loc[i, 'entity'].split(';'), data.loc[i, 'key_entity'].split(';'))
                if flag:
                    cnt_not_null_j += c_j
                    cnt_not_null_k += c_k
                    cnt_not_null_eq += c_eq
                else:
                    cnt_not_null_notin += 1
            else:
                cnt_not_null_keynull += 1
    else:
        cnt_null += 1
        if data.loc[i, 'negative'] == 0:
            cnt_null_0 += 1

print("not_null = ", cnt_not_null)
print("not_null and func = ", cnt_not_null_func)
print("其中key为null的 =", cnt_not_null_keynull)
print("重复部分不在key中的 =", cnt_not_null_notin)
print("大小相等且均包含在key中= ", cnt_not_null_eq)
print("其中更小的被包含在key中= ", cnt_not_null_j)
print("其中更大的被包含在key中= ", cnt_not_null_k)
print("null = ", cnt_null)
print("null and 0 = ", cnt_null_0)




data = pd.read_csv('data/test.csv', encoding='GB18030')
rows = data.shape[0]
cnt_not_null_func = 0
cnt_not_null = 0
cnt_null = 0
for i in range(0, rows):
    if isinstance(data.loc[i, 'entity'], str):
        cnt_not_null += 1
        if func(data.loc[i, 'entity'].split(';')):
            # print(data.loc[i, 'entity'].split(';'))
            cnt_not_null_func += 1
    else:
        cnt_null += 1

print("not_null = ", cnt_not_null)
print("not_null and func = ", cnt_not_null_func)
print("null = ", cnt_null)