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
# data = pd.read_csv('data/test.csv', encoding='GB18030')
# print(data.head())
rows = data.shape[0]
data_arr = data.values

lens_title = np.zeros(rows)
lens_text = np.zeros(rows)
list_title = []
list_text = []
list_all = []

for i in range(0, rows):
    tmp = 0
    if isinstance(data_arr[i, 1], str):
        lens_title[i] = len(data_arr[i, 1])
        list_title.append(len(data_arr[i, 1]))
        tmp += len(data_arr[i, 1])
    if isinstance(data_arr[i, 2], str):
        lens_text[i] = len(data_arr[i, 2])
        list_text.append(len(data_arr[i, 2]))
        tmp += len(data_arr[i, 2])
    if tmp:
        list_all.append(tmp)
    if i % 100:
        print(list_all[-1])

new_data = {
    "id": data['id'],
    "len_title": lens_title,
    "len_text": lens_text
}
new_data = pd.DataFrame(new_data)

new_data.to_csv('./length.csv', index=False)

'''
# lens_text = np.log(np.array(list_title))
# np.array(list_title)
plt.style.use('ggplot')
plt.hist(np.array(list_text),  # 绘图数据
         bins=500,
         color='steelblue',  # 指定填充色
         edgecolor='k',  # 指定直方图的边界色
         label='直方图')  # 为直方图呈现标签

# 去除图形顶部边界和右边界的刻度
plt.tick_params(top='off', right='off')
# 显示图例
plt.legend()
# 显示图形
plt.show()
'''


