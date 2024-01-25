import os
import re


def read_imdb(data_dir):
    texts = []
    labels = []
    for label_type in ['pos', 'neg']:
        label = 1 if label_type == 'pos' else 0
        dir_path = os.path.join(data_dir, label_type)
        for filename in os.listdir(dir_path):
            if filename.endswith('.txt'):
                with open(os.path.join(dir_path, filename), 'r', encoding='utf-8') as file:
                    text = file.read()
                    texts.append(text)
                    labels.append(label)
    return texts, labels


train_text, train_label = read_imdb('./aclImdb/train')
test_text, test_label = read_imdb('./aclImdb/test')

