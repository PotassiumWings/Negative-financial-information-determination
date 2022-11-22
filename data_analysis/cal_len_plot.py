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
data = pd.read_csv('my_data/length_train.csv', encoding='GB18030')

rows = data.shape[0]
data_arr = data.values

title_len = data_arr[:, 1]
text_len = data_arr[:, 2]

plt.style.use('ggplot')
plt.hist(title_len,  # 绘图数据
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



