import re
import matplotlib as mpl
from matplotlib import pyplot as plt


files = [
    "logs/log1123_200815.txt",
    "logs/log1124_022448.txt",
    "logs/log1124_040310.txt",
    "logs/log1124_111248.txt",
    "logs/log1124_140457.txt",
    "logs/log1124_193204.txt"
]

labels = [
    "lr 2e-6, BCEWithLogitsLoss",
    "lr 2e-6, CrossEntropyLoss",
    "lr 2e-6, prompt learning 好赞 差坏",
    "lr 2e-6, prompt learning 好 坏",
    "lr 2e-6, prompt learning 好 差",
    "roberta-base, lr 2e-6, BCEWithLogitsLoss"
]


mpl.rc("font", family="SimHei", size=24)
for i in range(len(files)):
    filename = files[i]
    f = open(filename, "r")
    print(f"Opening {filename}")
    x = []
    y = []
    for line in f:
        groups = re.findall(".*?INFO.*Ep.*?iter ([0-9]+).*val acc (0.[0-9]+).*", line)
        if groups is not None and len(groups) == 1:
            iter_id, val_acc = groups[0]
            iter_id = int(iter_id)
            val_acc = float(val_acc)
            x.append(iter_id)
            y.append(val_acc)
    plt.plot(x, y, label=labels[i])
plt.legend(labels)
plt.show()
