
# 大模型组寒假学习

## 任务一：数据处理
### 1. 读取imdb数据集并构建词表

>* 读取imdb数据集并构建词表（已有词表，但是还是希望能自己构建一个）
>* IMDB 数据集包含来自互联网电影数据库（IMDB）的 50 000 条严重两极分化的评论。数据集被分为用于训练的 25 000 条评论与用于测试的 25 000 条评论，训练集和测试集都包含 50% 的正面评论和 50% 的负面评论。
>* train_labels 和 test_labels 都是 0 和 1 组成的列表，其中 0代表负面（negative），1 代表正面（positive）
熟悉python相关语法，完成数据集的读取，与词表的构建。

 
* **数据集的下载**

可以从 https://ai.stanford.edu/~amaas/data/sentiment/ 下载并解压缩。
* **查看下载好的数据集**

点开下载好的数据集文件夹```aclImdb```点开后我们发现其中分为```test```和```train```两个文件夹，点开后又各自分为```pos```和```neg```两个文件夹，其中分别包含积极和消极的评论txt文本，我们需要依据这个进行程序设计。

![](https://img-blog.csdnimg.cn/img_convert/c797b94906028f11744eba9e48683a84.jpeg)

* **构建读取数据集的函数**

定义两个列表```texts```和```labels```分别存储评论文本和该评论的积极或消极标签，
```python
texts = []
labels = []
```
分别就进入```pos```和```neg```文件夹，积极标记为```1```，消极为```0```。使用```os.path.join```
函数进行路径的拼接，如果最后没有```/```符，该函数将会在最后自动补充```/```符
```python
for label_type in ['pos', 'neg']:
    label = 1 if label_type == 'pos' else 0
    dir_path = os.path.join(data_dir, label_type)  # 拼接路径
```
遍历文件夹下所有尾缀为```.txt```的文件。

1. 使用函数```os.listdir()```获取指定目录下的所有文件和子目录。对于```os.listdir(path)```，```path```为要获取文件和子目录列表的目录路径。如果不提供```path```参数，则默认为当前工作目录。该函数会返回一个包含指定目录下所有文件和子目录名称的列表。

2. 使用```string.endwith()```判断字符串是否以指定内容结尾。对于```string.endwith( str, start, end )```，参数```str```为必选，表示（字符串或元组）指定字符串或元素，元素```start```为可选，表示开始的索引，默认值```0```，元素```end```为可选，表示结束的索引，默认值```-1```
```python
for filename in os.listdir(dir_path):
    if filename.endswith('.txt'):
```
使用上下管理器以只读的方式打开文件，读入并加入元组

```python
with open(os.path.join(dir_path, filename), 'r', encoding='utf-8') as file:
    text = file.read()
    texts.append(text)
    labels.append(label)
```

读入部分的完整代码：

```python
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
```

* **什么是构建词表**

构建词表是指创建一个包含文本数据中所有独特单词的列表。在自然语言处理（NLP）和机器学习中，构建词表是一个重要的预处理步骤，它有助于将文本数据转换为可以被模型处理的形式。
构建词表的过程通常包括以下步骤：

1. **分词（Tokenization）：** 将文本数据分割成单个单词或标记的过程。

2. **去除停用词（Stopword Removal）：** 移除常见且对文本分析没有太多信息的停用词，例如“the”、“and”、“is”等。

3. **小写化（Lowercasing）：** 将所有单词转换为小写形式，以避免同一词汇以不同的大小写形式出现而被认为是不同的单词。

4. **移除标点符号和特殊字符：** 去除文本中的标点符号、特殊字符和数字，使得词表中只包含字母单词。

5. **去重：** 确保词表中每个单词只出现一次，以避免重复。                           

构建好词表后，每个单词通常会被分配一个唯一的索引或编号，以便在后续的处理中可以使用这些索引来表示文本数据。这种表示形式有助于将文本转化为数字形式，方便计算机进行处理，例如用于训练文本分类、语言模型等任务。

* **构建构建词表的函数**

声明一个计数器的类,用于记录各种单词的出现次数
```python
counter = Counter()
```
遍历train_text列表中的各个评论文本，对这些text进行转换。

1. 使用函数```string.lower()```，可以将```string```中的所有字母字符转换为小写形式。该函数接受一个字符串作为输入，并返回一个新的小写字符串。

2. 使用函数```maketrans() ```，可以定义字符串转换规则，对于 ```trantab = str.maketrans(intab,outtab,deltab) ```
>* 元素```intab```表示被代替字符，可以有多个，写在```""```之中，无则不写
>* 元素```outtab```表示被代替字符，可以有多个，写在```""```之中，无则不写，注意需要与```intab```中的对应且数量一致
>* 元素```deltab```表示需要删除的元素，格式同上。

此写法限Python3，Python2写法与之不同。

3. 使用函数```string.translate()```， 可以根据 ```maketrans()```函数给出的字符映射转换表转换字符串中的字符和删除字符。对于```test.translate(trantab)```表示使用转换规则```trantab```进行字符串的转换```text```字符串。

此写法限Python3，Python2写法与之不同。

4. 字符串```string.punctuation```表示所有的标点符号组成的字符串。

5.  使用函数```string.split()```，可以分割字符串，并返回一个列表。对于split(sep=None, maxsplit=-1)
>*  sep为切割，默认为空格
>*   maxsplit为切割次数，给值-1或者none，将会从左到右每一个sep切割一次
6. 用将列表里面的单词更新计数器
```python
for text in train_text:
	text = text.lower()  # 转换为小写
	text = text.translate(str.maketrans("", "", string.punctuation))  # 去除标点符号
	token = text.split()  # 分词
	counter.update(token)  # 计数（出现频率）
```
最后统计出现频率最高的单词，构成一个列表，构成词表

```python
voc = [word for word, _ in counter.most_common(max_vocab_size)]
```
* **读取imdb数据集并构建词表的总代码**

```python
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

```
### 2. 使用dataset，dataloader包装imdb数据集。
>* 学会使用dataset，dataloader功能。

* **Dataset,Dataloader是什么？**

**Dataset（数据集）:**   在机器学习中，数据集是模型训练和评估的基础。它是一个包含输入数据和相应标签（或目标）的集合。数据集可以分为训练集、验证集和测试集，用于模型的训练、调优和评估。

**Dataloader（数据加载器）:**   Dataloader是一个用于批量加载数据的工具，通常与Dataset一起使用。它可以帮助有效地管理和加载大规模的数据集，并将数据分批次提供给模型进行训练。Dataloader还可以提供数据的随机化和并行加载等功能，以优化训练效率。

* **为什么要了解Dataloader？**

​ 因为你的神经网络表现不佳的主要原因之一可能是由于数据不佳或理解不足。 因此，以更直观的方式理解、预处理数据并将其加载到网络中非常重要。​ 通常，我们在默认或知名数据集（如 MNIST 或 CIFAR）上训练神经网络，可以轻松地实现预测和分类类型问题的超过 90% 的准确度。 但是那是因为这些数据集组织整齐且易于预处理。 但是处理自己的数据集时，我们常常无法达到这样高的准确率。

* **为什么要定义自己的数据集？**
 定义自己的数据集通常是因为实际问题的数据可能不符合通用的标准格式，或者需要进行特殊的处理和预处理。在本问题中，我们需要自己构建数据集。
 
 * **构建构建imdb数据集的函数**
 构建imdb数据集需要覆盖原来三个函数，包括```__init__```，```__len__```和```__getitem__```。作用分别为初始化，返回数据集总大小，通过索引返回数据集中选定的样本。

```python
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
```
* **读取imdb数据集，构建词表，并构建数据集的总代码**

```python
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


```
