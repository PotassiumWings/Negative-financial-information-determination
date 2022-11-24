# 输入csv的title和text
# 输出分词和词频
# 先用分词工具、再查漏补缺手工设置正则表达式

import re
import csv
import jieba.analyse
import pandas as pd
import argparse


def regular(data, type="train"):
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
    title_file = type + '_title_reg.csv'
    text_file = type + '_text_reg.csv'

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


def filter(data, type="train"):
    if args.mode == 'jieba':
        title_cnt = {}
        text_cnt = {}
        # 加载停用词
        stopList = []
        if args.ifStop:
            stopList = [line.strip() for line in open(args.stopFile, encoding='utf-8')]

        # 正则过滤
        if args.regular:
            regular(data, type)

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
        title_file = type + 'title_seg_by_jieba.csv'
        text_file = type + 'text_seg_by_jieba.csv'
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
            regular(data)

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
        regular(data)

    else:
        # 加载停用词
        stopList = []
        if args.ifStop:
            stopList = [line.strip() for line in open(args.stopFile, encoding='utf-8')]

        # 正则过滤
        if args.regular:
            regular(data)

        # 分词 + 拼接
        write_path = args.clean_path + type + '_clean.csv'
        with open(write_path, "w", newline='', encoding='GB18030') as f:
            writer = csv.writer(f)
            if type == 'test':
                writer.writerow(["id", "title", "text", "entity"])
            else:
                writer.writerow(["id", "title", "text", "entity", "negative", "key_entity"])
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
                if type == 'test':
                    writer.writerow([line.id, new_title, new_text, line.entity])
                else:
                    if line.key_entity == "nan":
                        line.key_entity = None
                    writer.writerow([line.id, new_title, new_text, line.entity, line.negative, line.key_entity])
        f.close()


if __name__ == '__main__':
    USAGE = "usage:   python seg.py --path [file path] --clean_path [clean file path]"

    parser = argparse.ArgumentParser(USAGE)
    # edit according to your path
    parser.add_argument('--path', default='E:/Dataset/Machine Learning/finance/', help='source data path dir')
    parser.add_argument('--clean_path', default='', help='target clean data path')
    # normally useless
    parser.add_argument('--mode', default='QAQ')
    parser.add_argument('--topK', default=20)
    parser.add_argument('--stopFile', default='stopFile.txt', help='stop list file path')
    parser.add_argument('--ifStop', default=True, help='whether enable stop words filter')
    parser.add_argument('--regular', default=True, help='whether enable regular filter')

    args = parser.parse_args()
    train_path = args.path + "train.csv"
    test_path = args.path + "test.csv"
    train_data = pd.read_csv(train_path, header=0, engine='python', encoding='GB18030')
    test_data = pd.read_csv(test_path, header=0, engine='python', encoding='GB18030')
    train_data.dropna(axis=0, how="all", inplace=True)
    train_data.dropna(axis=0, how="all", inplace=True)
    train_data = train_data.astype(str)
    test_data = test_data.astype(str)

    filter(train_data, "train")
    filter(test_data, "test")

    print("good luck!")
