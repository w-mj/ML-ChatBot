import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import numpy as np
import jieba
import tensorflow.contrib.seq2seq as seq2seq
from utils import variable_summaries
import read_data


class seq2seqModel(object):
    def __init__(self):
        self.embedding_size = 300
        self.num_words = 85000  # 从文件读
        self.sentence_length = 32
        self.max_sentence_length = 50
        self.hidden_size = 128
        self.lstm_dims = 10
        self.word2index, self.index2word, embed, self.num_words, self.embedding_size = read_data.read_embedding()
        self.graph, self.loss, self.train_op, self.predict_output = self._model(embed)
        del embed

        summary_dir = 'summary'

        self.sess = tf.Session(graph=self.graph)

        self.writer = tf.summary.FileWriter(summary_dir, self.sess.graph)

        with self.graph.as_default():
            self.sess.run(tf.global_variables_initializer())
            try:
                saver = tf.train.Saver()
                saver.restore(self.sess, './model_save/seq2seq')
                print('load model.')
            except Exception:
                print('fail to read model.')

    def _preprocess_data(self, batch_x, batch_y):
        batch_x = list(map(str.split, batch_x))  # 把空格分隔的词转换成列表
        batch_y = map(str.split, batch_y)
        list(map(lambda x: x.reverse(), batch_x))
        # [x.reverse() for x in batch_x]
        max_length = self.max_sentence_length
        batch_x = map(lambda x: x[0: max_length] if len(x) > max_length else x, batch_x)
        batch_y = map(lambda x: x[0: max_length] if len(x) > max_length else x, batch_y)
        # batch_x = list(map(lambda x: ['GO'] + x + ['EOS'], batch_x))  # 在每句话前后添加开始和结束符
        batch_x = list(map(lambda x: x + ['EOS'], batch_x))
        batch_y = list(map(lambda x: x + ['EOS'], batch_y))
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

    def _cell(self, keep_prob):
        cell = rnn.LSTMCell(num_units=self.hidden_size)
        return rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)

    def _encoder(self, keep_prob, x_embedding, x_sequence_length, batch_size):
        num_layers = self.lstm_dims // 2
        cell_f = rnn.MultiRNNCell([self._cell(keep_prob) for _ in range(num_layers)])
        cell_b = rnn.MultiRNNCell([self._cell(keep_prob) for _ in range(num_layers)])
        # 计算encoder
        output, states = tf.nn.bidirectional_dynamic_rnn(cell_bw=cell_b, cell_fw=cell_f, inputs=x_embedding,
                                                         initial_state_bw=cell_b.zero_state(batch_size, tf.float32),
                                                         initial_state_fw=cell_f.zero_state(batch_size, tf.float32),
                                                         sequence_length=x_sequence_length)
        encoder_outputs = tf.concat(output, 2)

        encoder_state = []
        for layer_id in range(num_layers):
            encoder_state.append(states[0][layer_id])  # forward
            encoder_state.append(states[1][layer_id])  # backward
        encoder_state = tuple(encoder_state)

        return encoder_outputs, encoder_state
        # output, states = tf.nn.dynamic_rnn(cell=cell_f, inputs=x_embedding,
        #                                   initial_state=cell_f.zero_state(batch_size, tf.float32),
        #                                   sequence_length=x_sequence_length)
        # return output, states

    def _decoder(self, keep_prob, encoder_output, encoder_state, batch_size, scope, helper, reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            attention_states = encoder_output
            cell = rnn.MultiRNNCell([self._cell(keep_prob) for _ in range(self.lstm_dims)])
            attention_mechanism = seq2seq.BahdanauAttention(self.hidden_size, attention_states)  # attention
            decoder_cell = seq2seq.AttentionWrapper(cell, attention_mechanism,
                                                    attention_layer_size=self.hidden_size // 2)
            decoder_cell = rnn.OutputProjectionWrapper(decoder_cell, self.hidden_size, reuse=reuse,
                                                       activation=tf.nn.leaky_relu)
            decoder_initial_state = decoder_cell.zero_state(batch_size, tf.float32).clone(cell_state=encoder_state)

            output_layer = tf.layers.Dense(self.num_words,
                                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                           activation=tf.nn.leaky_relu)
            decoder = seq2seq.BasicDecoder(decoder_cell, helper, decoder_initial_state, output_layer=output_layer)
            output, _, _ = seq2seq.dynamic_decode(decoder, maximum_iterations=self.max_sentence_length,
                                                  impute_finished=True)

            # tf.summary.histogram('decoder', output)
        return output

    def _model(self, embed):
        graph = tf.Graph()
        with graph.as_default():
            embedding = tf.Variable(embed, trainable=False, name='embedding')  # 词向量
            lr = tf.placeholder(tf.float32, [], name='learning_rate')
            # 输入数据
            x_input = tf.placeholder(tf.int32, [None, None], name='x_input')  # 输入数据X
            x_sequence_length = tf.placeholder(tf.int32, [None], name='x_length')  # 输入数据每一条的长度
            x_embedding = tf.nn.embedding_lookup(embedding, x_input)  # 将输入的one-hot编码转换成向量
            y_input = tf.placeholder(tf.int32, [None, None], name='y_input')  # 输入数据Y
            y_sequence_length = tf.placeholder(tf.int32, [None], name='y_length')  # 每一个Y的长度
            y_embedding = tf.nn.embedding_lookup(embedding, y_input)  # 对Y向量化
            batch_size = tf.placeholder(tf.int32, [], name='batch_size')
            keep_prob = tf.placeholder(tf.float32, [], name='keep_prob')

            encoder_output, encoder_state = self._encoder(keep_prob, x_embedding, x_sequence_length, batch_size)

            training_helper = seq2seq.TrainingHelper(inputs=y_embedding, sequence_length=y_sequence_length)
            predict_helper = seq2seq.GreedyEmbeddingHelper(embedding, tf.fill([batch_size], self.word2index['GO']),
                                                           self.word2index['EOS'])
            train_output = self._decoder(keep_prob, encoder_output, encoder_state, batch_size, 'decode',
                                         training_helper)
            predict_output = self._decoder(keep_prob, encoder_output, encoder_state, batch_size, 'decode',
                                           predict_helper, True)

            # loss function
            training_logits = tf.identity(train_output.rnn_output, name='training_logits')
            predicting_logits = tf.identity(predict_output.rnn_output, name='predicting')

            # target = tf.slice(y_input, [0, 1], [-1, -1])
            # target = tf.concat([tf.fill([batch_size, 1], self.word2index['GO']), y_input], 1)
            target = y_input

            masks = tf.sequence_mask(y_sequence_length, dtype=tf.float32, name='mask')

            loss = seq2seq.sequence_loss(training_logits, target, masks)
            optimizer = tf.train.AdamOptimizer(lr)
            gradients = optimizer.compute_gradients(loss)
            capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if
                                grad is not None]
            train_op = optimizer.apply_gradients(capped_gradients)
            # predicting_logits = tf.nn.softmax(predicting_logits, axis=1)
            tf.summary.scalar('loss', loss)
            tf.summary.scalar('learning rate', lr)
            # tf.summary.tensor_summary('learning rate', lr)

        return graph, loss, train_op, predicting_logits

    def test_data(self, data):
        batch_size = 16
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
                             g.get_tensor_by_name('batch_size:0'): len(batch_x),
                             g.get_tensor_by_name('keep_prob:0'): 1
                             }
                loss, output, predict_output = self.sess.run(
                    [self.loss, g.get_tensor_by_name('training_logits:0'), self.predict_output],
                    feed_dict)
                # self.writer.add_summary(merged, i)
                if not printed:
                    objector = np.vectorize(lambda s: self.index2word[s])
                    output = np.argmax(output[0], 1)  # 按行取最大值
                    output = objector(output)
                    target = objector(batch_y[0])
                    target_in = objector(batch_x[0])
                    predict_output = np.argmax(predict_output[0], 1)  # 按行取最大值
                    predict_output = objector(predict_output)
                    print(
                        'input: {}\noutput\: {}\ntarget-input: {}\ntarget-output: {}\ntrain-output: {}\npredict-output: {}'.
                            format(x[0], y[0], ' '.join(target_in), ' '.join(target), ' '.join(output),
                                   ' '.join(predict_output)))
                    printed = True

                aver_loss += loss
        return aver_loss / batch_num

    def train(self, batch_size, learning_rate, max_epoch=1):
        train_data, valid_data, test_data = read_data.read_data_db()
        self.test_data(valid_data)
        _lr = learning_rate
        g = self.graph
        tr_batch_num = train_data.size // batch_size
        print("start training, max epoch: {0}, max batch: {1}".format(max_epoch, tr_batch_num))
        with self.sess.as_default():
            merged = tf.summary.merge_all()
            for epoch in range(max_epoch):
                _lr = _lr / (10 ** epoch)
                for i in range(tr_batch_num):
                    x, y = train_data.next_batch(batch_size)
                    assert len(x) == len(y)
                    batch_x, length_x, batch_y, length_y = self._preprocess_data(x, y)

                    feed_dict = {g.get_tensor_by_name('x_input:0'): batch_x,
                                 g.get_tensor_by_name('x_length:0'): length_x,
                                 g.get_tensor_by_name('y_input:0'): batch_y,
                                 g.get_tensor_by_name('y_length:0'): length_y,
                                 g.get_tensor_by_name('learning_rate:0'): _lr,
                                 g.get_tensor_by_name('batch_size:0'): len(length_x),
                                 g.get_tensor_by_name('keep_prob:0'): 0.8
                                 }
                    try:
                         loss_var, *_ = self.sess.run([ self.loss, self.train_op], feed_dict)
                        # self.writer.add_summary(me)
                    except tf.errors.InvalidArgumentError as e:
                        print(e)
                    if (i + 1) % 10 == 0:
                        print("training epoch {0}/{1}, learning rate = {2}, batch {3}/{4}, loss={5}"
                              .format(epoch + 1, max_epoch, _lr, i + 1, tr_batch_num, loss_var))

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
            print('saved')

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
    i = 8
    while True:
        # try:
        a.train(batch_size=32, learning_rate=1 / (10 ** i), max_epoch=1)
        # except Exception as e:
        #     print(e)
        i += 1
    # a.test('战狼56亿票房，旷世神作，中国第一')
    # a.test('人们提起网文都会说，第一部网络小说是痞子蔡的《第一次的亲密接触》。')
    # a.test('最后怎么解决的？')
    # a.train()
    a.save_model()

# reference https://github.com/AbrahamSanders/seq2seq-chatbot/
