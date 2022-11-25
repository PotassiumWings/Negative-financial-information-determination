import re

from matplotlib import pyplot as plt

labels = [
    ("lr 2e-6, BCEWithLogitsLoss", "logs/log1123_200815.txt"),
    ("lr 2e-6, CrossEntropyLoss", "logs/log1124_022448.txt"),
    ("lr 2e-6, prompt learning 2+2", "logs/log1124_040310.txt"),
    # ("lr 2e-6, prompt learning 1+1", "logs/log1124_111248.txt"),
    # ("lr 2e-6, prompt learning 1+1'", "logs/log1124_140457.txt"),
    ("lr 2e-6, BCEWithLogitsLoss, roberta-base", "logs/log1124_193204.txt"),
    ("lr 6e-6, prompt learning 2+2", "logs/log1125_010246.txt"),
    # ("lr 3e-6, prompt learning 2+2 AdamW+SGD", "logs/log1125_034622.txt"),
    # ("lr 3e-6, prompt learning 2+2 clean", "logs/log1125_110908.txt"),
    ("lr 5e-6, prompt learning - roberta-large-wwm", "logs/log1125_141628.txt"),
    ("lr 5e-6, prompt learning - xlm-roberta-large", "logs/log1125_150142.txt"),
    ("lr 6e-6, prompt learning clean_stop_", "logs/log1125_171329.txt")
]
pure_labels = [_[0] for _ in labels]


def main():
    plt.figure(figsize=(20, 20))
    for i in range(len(labels)):
        label, filename = labels[i]

        def process_f(filename, encoding="utf-8"):
            f = open(filename, "r", encoding=encoding, errors=None)
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
            plt.plot(x, y, label=label)

        try:
            process_f(filename)
        except Exception:
            process_f(filename, "")

    leg = plt.legend(pure_labels)
    for line in leg.get_lines():
        line.linewidth = 4
    plt.savefig("analysis.png")

main()
