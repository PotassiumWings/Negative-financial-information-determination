# 输入csv的title和text
# 输出分词和词频
# 先用分词工具、再查漏补缺手工设置正则表达式

import jieba
import re
import csv
import jieba.analyse
import pandas as pd
import argparse

USAGE = "usage:   python seg.py --path [file path] --seg [algorithm]"

parser = argparse.ArgumentParser(USAGE)
parser.add_argument('--path', default='E:/Dataset/Machine Learning/finance/train.csv')
parser.add_argument('--mode', default='tfidf')
parser.add_argument('--topK', default=20)
parser.add_argument('--ifStop', default=False)
parser.add_argument('--stopFile', default='stopFile.txt')
parser.add_argument('--regular', default=False)
parser.add_argument('--clean', default='clean.csv')

args = parser.parse_args()
data = pd.read_csv(args.path, header=0, engine='python', encoding='GB18030').astype(str)
data.dropna(axis=0, how="all", inplace=True)


def regular():
    title_res = {0: [],
                 1: [],
                 2: [],
                 3: [],
                 4: [],
                 5: [],
                 6: []
                 }
    text_res = {0: [],
                1: [],
                2: [],
                3: [],
                4: [],
                5: [],
                6: []
                }
    # 正则匹配式
    patterns = ["<[^>]+>",
                "https?://[_a-zA-Z0-9./-]+",
                "[_a-zA-Z0-9]+@[_a-zA-Z0-9]+.com",
                "#[^#]+#",
                "{IMG:[0-9]+}",
                "((20|19)[0-9]{2}[^0-9]?[0-9]{1,2}[^0-9]?[0-9]{1,2}\u65E5?)",
                "\s"]
    for _, line in data.iterrows():
        for i in range(len(patterns)):
            l = re.findall(patterns[i], line.title)
            title_res[i].extend(l)
            line.title = re.sub(patterns[i], "", line.title)

            l = re.findall(patterns[i], line.text)
            text_res[i].extend(l)
            line.text = re.sub(patterns[i], "", line.text)

    # 输出到csv
    title_file = 'title_reg.csv'
    text_file = 'text_reg.csv'

    with open(title_file, "w", newline='', encoding='GB18030') as f:
        w = csv.writer(f)
        for _, v in title_res.items():
            w.writerow(v)
    f.close()

    with open(text_file, "w", newline='', encoding='GB18030') as f:
        w = csv.writer(f)
        for _, v in text_res.items():
            w.writerow(v)
    f.close()


if args.mode == 'jieba':
    title_cnt = {}
    text_cnt = {}
    # 加载停用词
    stopList = []
    if args.ifStop:
        stopList = [line.strip() for line in open(args.stopFile, encoding='utf-8')]

    # 正则过滤
    if args.regular:
        regular()

    # 分词 + 统计
    for index, line in data.iterrows():
        title_cut = jieba.cut(line.title, cut_all=False)
        text_cut = jieba.cut(line.text, cut_all=False)
        for word in title_cut:
            if args.ifStop and word in stopList:
                continue
            title_cnt[word] = title_cnt.get(word, 0) + 1
        for word in text_cut:
            if args.ifStop and word in stopList:
                continue
            text_cnt[word] = text_cnt.get(word, 0) + 1

    # 输出到csv
    title_file = 'title_seg_by_jieba.csv'
    text_file = 'text_seg_by_jieba.csv'
    with open(title_file, "w", newline='', encoding='GB18030') as f:
        writer = csv.writer(f)
        for key, value in title_cnt.items():
            writer.writerow([key, value])
    f.close()

    with open(text_file, "w", newline='', encoding='GB18030') as f:
        writer = csv.writer(f)
        for key, value in text_cnt.items():
            writer.writerow([key, value])
    f.close()

# 直接基于TF-IDF做关键词抽取，默认返回20个TF/IDF权重最大的关键词
elif args.mode == 'tfidf':
    # 正则过滤
    if args.regular:
        regular()

    title_txt = ""
    text_txt = ""
    for index, line in data.iterrows():
        title_txt += line.title
        text_txt += line.text
    if args.ifStop:
        jieba.analyse.set_stop_words(args.stopFile)
    title_tags = jieba.analyse.extract_tags(title_txt)
    text_tags = jieba.analyse.extract_tags(text_txt)
    print(title_tags)
    print("+++++++++++++++++++++++++")
    print(text_tags)

elif args.mode == 'reg':
    # 只统计
    regular()

else:
    # 加载停用词
    stopList = []
    if args.ifStop:
        stopList = [line.strip() for line in open(args.stopFile, encoding='utf-8')]

    # 正则过滤
    if args.regular:
        regular()

    # 分词 + 拼接
    with open(args.clean, "w", newline='', encoding='GB18030') as f:
        writer = csv.writer(f)
        for index, line in data.iterrows():
            title_cut = jieba.cut(line.title, cut_all=False)
            text_cut = jieba.cut(line.text, cut_all=False)
            new_title = ""
            new_text = ""
            for word in title_cut:
                if args.ifStop and word in stopList:
                    continue
                new_title = new_title + word
            for word in text_cut:
                if args.ifStop and word in stopList:
                    continue
                new_text = new_text + word
            writer.writerow([line.id, new_title, new_text, line.entity, line.negative, line.key_entity])
    f.close()
