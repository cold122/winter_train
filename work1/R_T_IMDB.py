import os
import string
from collections import Counter


def read_imdb(data_dir):
    texts = []
    labels = []
    for label_type in ['pos', 'neg']:
        label = 1 if label_type == 'pos' else 0
        dir_path = os.path.join(data_dir, label_type)  # 拼接路径
        for filename in os.listdir(dir_path):
            if filename.endswith('.txt'):
                with open(os.path.join(dir_path, filename), 'r', encoding='utf-8') as file:
                    text = file.read()
                    texts.append(text)
                    labels.append(label)
    return texts, labels


def build_vocab(max_vocab_size=None):
    counter = Counter()
    for text in train_text:
        text = text.lower()  # 转换为小写
        text = text.translate(str.maketrans("", "", string.punctuation))  # 去除标点符号
        token = text.split()  # 分词
        counter.update(token)  # 计数（出现频率）
    voc = [word for word, _ in counter.most_common(max_vocab_size)]  # 提取出现频率最高的
    return voc


if __name__ == "__main__":

    train_text, train_label = read_imdb('E:/pyprogram/winter_train/aclImdb/train')
    test_text, test_label = read_imdb('E:/pyprogram/winter_train/aclImdb/test')

    vocab = build_vocab(max_vocab_size=10000)

    # print("Vocabulary:", vocab[:20])
    # print("Vocabulary size:", len(vocab))


