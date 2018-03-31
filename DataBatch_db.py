'''
用于数据库的DataBatch，next_batch方法与DataBatch兼容。
此类中的数据不保存在内存中，而是当next_batch方法调用时在数据库中查找
'''

import numpy as np
import logging


class DataBatch_db(object):
    def __init__(self, db, id_list=None, shuffle=False):
        self._cursor = db.cursor()
        if id_list is not None:
            self._size = len(id_list)
            self._ids = id_list
        else:
            self._size = self._cursor.execute('select count(*) as num from conversation')[0][0]
            self._ids = range(self._size)

        self._ids = np.asarray(self._ids)
        if shuffle:
            index = np.random.permutation(self._size)
            self._ids = self._ids[index]
        self._shuffle = shuffle
        self._index = 0

    def next_batch(self, batch_size):
        index = self._index
        size = self._size
        if index + batch_size <= size:
            query = tuple(self._ids[index: index + batch_size])
            self._index += batch_size
        else:
            query = tuple(self._ids[index:] + self._ids[0: index + batch_size - size])
            self._index = index + batch_size - size
            if self._shuffle:
                index = np.random.permutation(self._size)
                self._ids = self._ids[index]

        value = self._cursor.execute('SELECT * from conversation where rowid in %s' % str(query)).fetchall()
        value = np.asarray(value)
        if len(value) != batch_size:
            logging.warning('number of result is less than expect while query the database with key={}'.format(str(query)))
        # TODO: 偶尔会出现查询数据库的返回个数比query少1的情况
        return value[:, 0].tolist(), value[:, 1].tolist()


    @property
    def size(self):
        return self._size

    def __del__(self):
        self._cursor.close()
