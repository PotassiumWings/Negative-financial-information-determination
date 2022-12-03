import numpy as np
import pandas as pd

filename = 'submission'
data = pd.read_csv('./' + filename + '.csv')
rows = data.shape[0]

print(1)

def func(en_list):
    list_new = []
    for en in en_list:
        if en.replace('?', '') != en:
            list_new.append(en)
    for en in en_list:
        flag = True
        for tmp in list_new:
            if en == tmp or tmp.replace('?', '') == en:
                flag = False
                break
        if flag:
            list_new.append(en)
    return list_new

for i in range(0, rows):
    if data.loc[i, 'negative'] == 1 and isinstance(data.loc[i, 'key_entity'], str):
        str1 = data.loc[i, 'key_entity']
        k_entity = str1.split(';')
        k_entity_new = func(k_entity)
        str_new = ""
        if len(k_entity_new) != 0:
            for en in k_entity_new:
                str_new += en + ';'
        str2 = str_new[0:-1]
        if str1 != str2:
            print("str1 = " + str1)
            print("str2 = " + str2)
            data.loc[i, 'key_entity'] = str2

data.to_csv('./' + filename + '_filter.csv', index=None)


