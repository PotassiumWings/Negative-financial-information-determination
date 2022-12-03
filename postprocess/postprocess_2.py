import numpy as np
import pandas as pd
from collections import Counter

def list_equal(l1, l2):
    return Counter(l1) == Counter(l2)


def process(l1):
    l2 = sorted(l1, key=lambda t: len(t), reverse=True)
    l3 = []
    for word in l2:
        flag = True
        for word_2 in l3:
            if word in word_2:
                flag = False
                break
        if flag:
            l3.append(word)
    return l3

data = pd.read_csv('submission.csv')
rows = data.shape[0]
for i in range(0, rows):
    if data.loc[i, 'negative'] == 1 :
        if isinstance(data.loc[i, 'key_entity'], str):
            res = data.loc[i, 'key_entity'].split(';')
            k_entity_new = process(res)
            str_new = ""
            if len(k_entity_new) != 0:
                for en in k_entity_new:
                    str_new += en + ';'
            str2 = str_new[0:-1]
            data.loc[i, 'key_entity'] = str2
        else:
            print(i)

data.to_csv('submission.csv', index=None)



