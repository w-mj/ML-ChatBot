import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import numpy as np
import sqlite3
import jieba
import tensorflow.contrib.seq2seq as seq2seq
from sklearn.model_selection import train_test_split
from DataBatch import DataBatch
from DataBatch_db import DataBatch_db


class seq2seqModel(object):
    def __init__(self):
        self.embedding_size = 128
        self.num_words = 85000
        self.sentence_length = 32
        self.max_sentence_length = 50
        self.word2index, self.index2word = self._read_dict()
        self.graph, self.loss, self.train_op, self.predict_output = self._model()
        self.sess = tf.Session(graph=self.graph)
        self.separator = None
        self.db = None
        with self.graph.as_default():
            self.sess.run(tf.global_variables_initializer())
            try:
                saver = tf.train.Saver()
                saver.restore(self.sess, './model_save/seq2seq')
            except (ValueError, tf.errors.NotFoundError):
                print('fail to read model.')
            embedding_saver = tf.train.Saver({'embeddings': self.graph.get_tensor_by_name('embedding:0')})
            embedding_saver.restore(self.sess, './word_embedding/embed')

    @staticmethod
    def _read_dict():
        word2index = {}
        index2word = {}
        with open('word_dictionary.txt') as f:
            for line in f:
                w, i = line.split()
                word2index[w] = int(i)
                index2word[int(i)] = w
        return word2index, index2word

    @staticmethod
    def _read_data_pylist():
        with open('train_data.txt', encoding='utf-8') as f:
            data = eval(f.read())
            print("load train_data.txt, {0} items altogether.".format(len(data)))
        x, y = list(map(list, zip(*data)))  # 拆分x和y
        x, _ = list(map(list, zip(*x)))  # 去掉不明数字
        y, _ = list(map(list, zip(*y)))
        x = list(map(lambda t: t[: -1], x))
        y = list(map(lambda t: t[: -1], y))
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
        x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2)
        return DataBatch(x_train, y_train), DataBatch(x_valid, y_valid), DataBatch(x_test, y_test)

    def _read_data_db(self):
        self.db = db = sqlite3.connect('data_separated.db')
        size = db.execute('SELECT count (*) AS num FROM conversation').fetchall()[0][0]
        print('open database, {} items altogether'.format(size))
        ids = np.random.random_integers(0, size, [size])
        return DataBatch_db(db, ids[0: int(size * 0.6)]), \
               DataBatch_db(db, ids[int(size * 0.6): int(size * 0.8)]), \
               DataBatch_db(db, ids[int(size * 0.8):]),

    def _preprocess_data(self, batch_x, batch_y):
        batch_x = list(map(str.split, batch_x))  # 把空格分隔的词转换成列表
        batch_y = list(map(str.split, batch_y))
        max_length = self.max_sentence_length
        batch_x = list(map(lambda x: x[0: max_length] if len(x) > max_length else x, batch_x))
        batch_y = list(map(lambda x: x[0: max_length] if len(x) > max_length else x, batch_y))
        batch_x = list(map(lambda x: ['GO'] + x + ['EOS'], batch_x))  # 在每句话前后添加开始和结束符
        batch_y = list(map(lambda x: ['GO'] + x + ['EOS'], batch_y))
        length_x = list(map(len, batch_x))  # 获得每句话的长度
        length_y = list(map(len, batch_y))
        max_lx = max(length_x)  # 最大长度
        max_ly = max(length_y)
        batch_x = list(map(lambda x: x + (['PAD'] * (max_lx - len(x))), batch_x))  # 小于最大长度的句子用PAD补足
        batch_y = list(map(lambda x: x + (['PAD'] * (max_ly - len(x))), batch_y))
        batch_x = np.array(batch_x)  # 转numpy数组
        batch_y = np.array(batch_y)
        batch_x = np.vectorize(lambda x: self.word2index.get(x, self.word2index['UNK']))(batch_x)  # 字符映射index
        batch_y = np.vectorize(lambda x: self.word2index.get(x, self.word2index['UNK']))(batch_y)
        return batch_x, length_x, batch_y, length_y

    def _model(self):
        graph = tf.Graph()
        with graph.as_default():
            embedding = tf.Variable(np.zeros(shape=[self.num_words, self.embedding_size], dtype=np.float32),
                                    trainable=False, name='embedding')  # 词向量
            lr = tf.placeholder(tf.float32, [], name='learning_rate')
            # 输入数据
            x_input = tf.placeholder(tf.int32, [None, None], name='x_input')  # 输入数据X
            x_sequence_length = tf.placeholder(tf.int32, [None], name='x_length')  # 输入数据每一条的长度
            x_embedding = tf.nn.embedding_lookup(embedding, x_input)  # 将输入的one-hot编码转换成向量
            y_input = tf.placeholder(tf.int32, [None, None], name='y_input')  # 输入数据Y
            y_sequence_length = tf.placeholder(tf.int32, [None], name='y_length')  # 每一个Y的长度
            y_embedding = tf.nn.embedding_lookup(embedding, y_input)  # 对Y向量化
            batch_size = tf.placeholder(tf.int32, [], name='batch_size')
            # batch_size = tf.shape(x_input)[0]
            # 使用gru代替LSTM, 4层cell堆叠
            encoder_cell = rnn.MultiRNNCell([rnn.GRUCell(128, activation=tf.tanh) for _ in range(4)])
            decoder_cell = rnn.MultiRNNCell([rnn.GRUCell(128, activation=tf.tanh) for _ in range(4)])
            # 计算encoder
            output, encoder_state = tf.nn.dynamic_rnn(cell=encoder_cell, inputs=x_embedding,
                                                      initial_state=encoder_cell.zero_state(batch_size, tf.float32),
                                                      sequence_length=x_sequence_length)

            output_layer = tf.layers.Dense(self.num_words,
                                           kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
            # 定义training decoder
            training_helper = seq2seq.TrainingHelper(inputs=y_embedding, sequence_length=y_sequence_length)
            training_decoder = seq2seq.BasicDecoder(decoder_cell, training_helper, encoder_state, output_layer)
            # impute_finish 标记为True时，序列读入<eos>后不再进行计算，保持state不变并且输出全0
            training_output, _, _ = seq2seq.dynamic_decode(training_decoder,
                                                           # 加上<GO>和<EOS>
                                                           maximum_iterations=self.max_sentence_length + 2,
                                                           impute_finished=True)

            # predict decoder
            predict_helper = seq2seq.GreedyEmbeddingHelper(embedding, tf.fill([batch_size], self.word2index['GO']),
                                                           self.word2index['EOS'])
            predict_decoder = seq2seq.BasicDecoder(decoder_cell, predict_helper, encoder_state, output_layer)
            predict_output, _, _ = seq2seq.dynamic_decode(predict_decoder,
                                                          maximum_iterations=self.max_sentence_length + 2,
                                                          impute_finished=True)

            # loss function
            training_logits = tf.identity(training_output.rnn_output, name='training_logits')
            predicting_logits = tf.identity(predict_output.rnn_output, name='predicting')

            masks = tf.sequence_mask(y_sequence_length, dtype=tf.float32, name='mask')

            with tf.variable_scope('optimization'):
                loss = seq2seq.sequence_loss(training_logits, y_input, masks)
                optimizer = tf.train.AdamOptimizer(lr)
                gradients = optimizer.compute_gradients(loss)
                capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if
                                    grad is not None]
                train_op = optimizer.apply_gradients(capped_gradients)

        return graph, loss, train_op, predicting_logits

    def test_data(self, data):
        batch_size = 64
        batch_num = 10
        g = self.graph
        aver_loss = 0
        printed = False
        with self.sess.as_default():
            for i in range(batch_num):
                x, y = data.next_batch(batch_size)
                batch_x, length_x, batch_y, length_y = self._preprocess_data(x, y)
                feed_dict = {g.get_tensor_by_name('x_input:0'): batch_x,
                             g.get_tensor_by_name('x_length:0'): length_x,
                             g.get_tensor_by_name('y_input:0'): batch_y,
                             g.get_tensor_by_name('y_length:0'): length_y,
                             g.get_tensor_by_name('batch_size:0'): len(batch_x)
                             }
                loss, output, predict_output = self.sess.run([self.loss, g.get_tensor_by_name('training_logits:0'),
                                                              self.predict_output], feed_dict)
                if not printed:
                    output = np.argmax(output[0], 1)  # 按行取最大值
                    output = np.vectorize(lambda s: self.index2word[s])(output)
                    predict_output = np.argmax(predict_output[0], 1)  # 按行取最大值
                    predict_output = np.vectorize(lambda s: self.index2word[s])(predict_output)
                    print('input: {0}\ntarget-output: {1}\ntrain-output: {2}\npredict-output: {3}'.
                          format(x[0], y[0], ''.join(output), ''.join(predict_output)))
                    printed = True

                aver_loss += loss
        return aver_loss / batch_num

    def train(self, batch_size=64, _lr=0.002, max_epoch=1):
        train_data, valid_data, test_data = self._read_data_db()
        self.test_data(valid_data)

        g = self.graph
        tr_batch_num = train_data.size // batch_size
        print("start training, max epoch: {0}, max batch: {1}".format(max_epoch, tr_batch_num))
        with self.sess.as_default():
            for epoch in range(max_epoch):
                for i in range(tr_batch_num):
                    x, y = train_data.next_batch(batch_size)
                    assert len(x) == len(y)
                    batch_x, length_x, batch_y, length_y = self._preprocess_data(x, y)

                    feed_dict = {g.get_tensor_by_name('x_input:0'): batch_x,
                                 g.get_tensor_by_name('x_length:0'): length_x,
                                 g.get_tensor_by_name('y_input:0'): batch_y,
                                 g.get_tensor_by_name('y_length:0'): length_y,
                                 g.get_tensor_by_name('learning_rate:0'): _lr,
                                 g.get_tensor_by_name('batch_size:0'): len(length_x)
                                 }
                    try:
                        loss_var, *_ = self.sess.run([self.loss, self.train_op], feed_dict)
                    except tf.errors.InvalidArgumentError as e:
                        print(e)
                    if (i + 1) % 10 == 0:
                        print("training epoch {0}/{1}, batch {2}/{3}, loss={4}"
                              .format(epoch + 1, max_epoch, i + 1, tr_batch_num, loss_var))

                    if (i + 1) % 100 == 0:
                        loss_var = self.test_data(valid_data)
                        print("valid result: loss={0}".format(loss_var))
                        self.save_model()
            loss_var = self.test_data(test_data)
            print('train finished, test tesult: loss={0}'.format(loss_var))

    def save_model(self):
        with self.graph.as_default():
            saver = tf.train.Saver()
            saver.save(self.sess, './model_save/seq2seq')

    def test(self, sentence):
        question = jieba.cut(sentence)  # 分词
        print(question)
        question = [['GO'] + question + ['EOS']]
        question = np.array(question)
        question = np.vectorize(lambda x: self.word2index.get(x, self.word2index['UNK']))(question)
        print(question)
        with self.sess.as_default():
            g = self.graph
            answer, *_ = self.sess.run([self.predict_output], feed_dict={
                g.get_tensor_by_name('x_input:0'): question,
                g.get_tensor_by_name('x_length:0'): [len(question[0])],
                g.get_tensor_by_name('batch_size:0'): 1
            })
        answer = answer[0]  # 去掉batch层
        print(answer[:10, :10])
        answer = np.argmax(answer, 1)  # 按行取最大值
        answer = np.vectorize(lambda x: self.index2word[x])(answer)
        print(answer)


if __name__ == '__main__':
    a = seq2seqModel()
    a.train()
    # a.test('战狼56亿票房，旷世神作，中国第一')
    # a.test('人们提起网文都会说，第一部网络小说是痞子蔡的《第一次的亲密接触》。')
    # a.test('最后怎么解决的？')
    # a.train()
    a.save_model()
