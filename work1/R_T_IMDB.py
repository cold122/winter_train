import os
import string
from torch.utils.data import Dataset, DataLoader
from collections import Counter


class IMDBDataset(Dataset):
    def __init__(self, text_in, label_in, vocab_in):
        self.text = text_in
        self.label = label_in
        self.vocab = vocab_in

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        text = self.text[idx]
        label = self.text[idx]

        # 分词和数字化处理
        numericalized_tokens = [self.vocab[token] for token in text]

        return {'text': numericalized_tokens, 'label': label}


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

    train_dataset = IMDBDataset(train_text, train_label, vocab)
    test_dataset = IMDBDataset(test_text, test_label, vocab)

    # 创建数据加载器
    batch_size = 64
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

