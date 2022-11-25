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


files = ['clean_stop_train', 'clean_stop_test',
         'clean_train', 'clean_test',
         'half_clean_train', 'half_clean_test',
         'nostop_clean_train', 'nostop_clean_test']
for file in files:
    data = pd.read_csv('filter_res_combine/' + file + '_combine.csv', encoding='GB18030')
    rows = data.shape[0]

    lens_text = np.zeros(rows)

    for i in range(0, rows):
        tmp = 0
        if isinstance(data.loc[i, 'text'], str):
            lens_text[i] = len(data.loc[i, 'text'])

    plt.style.use('ggplot')
    plt.hist(lens_text,
             bins=500,
             color='steelblue',
             edgecolor='k',
             label=file)

    plt.tick_params(top='off', right='off')
    plt.legend()
    plt.show()


