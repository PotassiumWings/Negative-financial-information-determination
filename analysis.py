import re
import matplotlib as mpl
from matplotlib import pyplot as plt


files = [
    "logs/log1123_200815.txt",
    "logs/log1124_022448.txt",
    "logs/log1124_040310.txt",
    "logs/log1124_111248.txt",
    "logs/log1124_140457.txt",
    "logs/log1124_193204.txt",
    "logs/log1125_010246.txt",
    "logs/log1125_034622.txt",
    "logs/log1125_110908.txt",
]

labels = [
    "lr 2e-6, BCEWithLogitsLoss",
    "lr 2e-6, CrossEntropyLoss",
    "lr 2e-6, prompt learning 2+2",
    "lr 2e-6, prompt learning 1+1",
    "lr 2e-6, prompt learning 1+1'",
    "roberta-base, lr 2e-6, BCEWithLogitsLoss",
    "lr 6e-6, prompt learning 2+2",
    "lr 3e-6, prompt learning 2+2 AdamW+SGD",
    "lr 3e-6, prompt learning 2+2 clean"
]


plt.figure(figsize=(20, 20))
for i in range(len(files)):
    filename = files[i]
    f = open(filename, "r")
    print(f"Opening {filename}")
    x = []
    y = []
    last_iter = -1
    pretrain_iter = 0
    for line in f:
        groups = re.findall(".*?INFO.*Ep.*?iter ([0-9]+).*val acc (0.[0-9]+).*", line)
        if groups is not None and len(groups) == 1:
            iter_id, val_acc = groups[0]
            iter_id = int(iter_id)
            val_acc = float(val_acc)
            if iter_id < last_iter:
                pretrain_iter = last_iter
            last_iter = iter_id

            x.append(iter_id + pretrain_iter)
            y.append(val_acc)
    plt.plot(x, y, label=labels[i])
leg = plt.legend(labels)
for line in leg.get_lines():
    line.linewidth = 4
plt.savefig("analysis.png")
