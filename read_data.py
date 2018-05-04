import io
from sklearn.model_selection import train_test_split
from DataBatch import DataBatch
from DataBatch_db import DataBatch_db
import sqlite3
import numpy as np


def read_embedding():
    fin = io.open('word_embed_clean.vec', 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = []
    word2index = {}
    index2word = {}
    for i, line in enumerate(fin):
        tokens = line.strip().split()
        data.append(list(map(float, tokens[1:])))
        assert len(data[-1]) == d
        word2index[tokens[0]] = i
        index2word[i] = tokens[0]
    return word2index, index2word, np.asarray(data, dtype=np.float32), n, d


def read_dict():
    word2index = {}
    index2word = {}
    with open('word_dictionary.txt') as f:
        for line in f:
            w, i = line.split()
            word2index[w] = int(i)
            index2word[int(i)] = w
    return word2index, index2word


def read_data_pylist():
    with open('train_data.txt', encoding='utf-8') as f:
        data = eval(f.read())
        print("load train_data.txt, {0} items altogether.".format(len(data)))
    x, y = list(map(list, zip(*data)))  # 拆分x和y
    x, _ = list(map(list, zip(*x)))  # 去掉不明数字
    y, _ = list(map(list, zip(*y)))
    x = list(map(lambda t: t[: -1], x))
    y = list(map(lambda t: t[: -1], y))
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.01)
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.01)
    return DataBatch(x_train, y_train), DataBatch(x_valid, y_valid), DataBatch(x_test, y_test)


def read_data_db():
    db = sqlite3.connect('data_separated.db')
    size = db.execute('SELECT count (*) AS num FROM conversation').fetchall()[0][0]
    print('open database, {} items altogether'.format(size))
    ids = np.random.permutation(size)
    return DataBatch_db(db, ids[0: int(size * 0.6)]), \
           DataBatch_db(db, ids[int(size * 0.6): int(size * 0.8)]), \
           DataBatch_db(db, ids[int(size * 0.8):])
