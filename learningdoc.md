# 大模型组寒假学习

## 任务一：数据处理
### 1. 读取imdb数据集并构建词表
```
读取imdb数据集并构建词表（已有词表，但是还是希望能自己构建一个）
IMDB 数据集包含来自互联网电影数据库（IMDB）的 50 000 条严重两极分化的评论。数据集被分为用于训练的 25 000 条评论与用于测试的 25 000 条评论，训练集和测试集都包含 50% 的正面评论和 50% 的负面评论。
train_labels 和 test_labels 都是 0 和 1 组成的列表，其中 0代表负面（negative），1 代表正面（positive）
熟悉python相关语法，完成数据集的读取，与词表的构建。
```

 
* **数据集的下载**

可以从 https://ai.stanford.edu/~amaas/data/sentiment/ 下载并解压缩。
* **查看下载好的数据集**

点开下载好的数据集文件夹```aclImdb```点开后我们发现其中分为```test```和```train```两个文件夹，点开后又各自分为```pos```和```neg```两个文件夹，其中分别包含积极和消极的评论txt文本，我们需要依据这个进行程序设计。

![](https://pic.imgdb.cn/item/65b4c775871b83018a695cee.jpg)

* **构建读取数据集的函数**

定义两个列表```texts```和```labels```分别存储评论文本和该评论的积极或消极标签，
```
texts = []
labels = []
```
分别就进入```pos```和```neg```文件夹，积极标记为```1```，消极为```0```。使用```os.path.join```
函数进行路径的拼接，如果最后没有```/```符，该函数将会在最后自动补充```/```符
```
for label_type in ['pos', 'neg']:
    label = 1 if label_type == 'pos' else 0
    dir_path = os.path.join(data_dir, label_type)  # 拼接路径
```
遍历文件夹下所有尾缀为```.txt```的文件。

使用函数```os.listdir()```获取指定目录下的所有文件和子目录。对于```os.listdir(path)```，```path```为要获取文件和子目录列表的目录路径。如果不提供```path```参数，则默认为当前工作目录。该函数会返回一个包含指定目录下所有文件和子目录名称的列表。

使用```endwith()```判断字符串是否以指定内容结尾。对于```string.endwith( str, start, end )```，参数```str```为必选，表示（字符串或元组）指定字符串或元素，元素```start```为可选，表示开始的索引，默认值```0```，元素```end```为可选，表示结束的索引，默认值```-1```
```
for filename in os.listdir(dir_path):
    if filename.endswith('.txt'):
```
使用上下管理器以只读的方式打开文件，读入并加入元组
```
with open(os.path.join(dir_path, filename), 'r', encoding='utf-8') as file:
    text = file.read()
    texts.append(text)
    labels.append(label)
```
读入部分的完整代码：
```
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














