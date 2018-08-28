import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import numpy as np
import jieba
import tensorflow.contrib.seq2seq as seq2seq
import read_data
import time
import vocabulary


class seq2seqModel(object):
    def __init__(self):
        self.max_sentence_length = 35
        self.rnn_hidden_size = 128
        self.lstm_dims = 4
        self.voca = vocabulary.Vocabulary()
        self.embed_np = read_data.read_embedding()
        self.graph, self.loss, self.train_op, self.predict_output = self._model()

        # summary_dir = 'summary'
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=self.graph, config=config)
        # self.writer = tf.summary.FileWriter(summary_dir, self.sess.graph)
        with self.graph.as_default():
            self.sess.run(tf.global_variables_initializer())
            try:
                saver = tf.train.Saver()
                saver.restore(self.sess, './model_save/seq2seq')
                print('load model.')
            except Exception:
                print('fail to load model.')

    def _preprocess_data(self, batch_x, batch_y):
        # list(map(lambda x: x.reverse(), batch_x))
        # [x.reverse() for x in batch_x]
        max_length = self.max_sentence_length
        batch_x = [x if len(x) < max_length - 1 else x[:max_length - 1] for x in batch_x]
        batch_y = [x if len(x) < max_length - 1 else x[:max_length - 1] for x in batch_y]
        batch_x = [self.voca.prepare_text(x) for x in batch_x]
        batch_y = [self.voca.prepare_text(x) for x in batch_y]
        # batch_x = list(map(lambda x: ['GO'] + x + ['EOS'], batch_x))  # 在每句话前后添加开始和结束符
        length_x = [len(x) for x in batch_x]  # 获得每句话的长度
        length_y = [len(x) for x in batch_y]
        max_lx = max(length_x)  # 最大长度
        max_ly = max(length_y)
        batch_x = [x + ([self.voca.PAD] * (max_lx - len(x))) for x in batch_x]  # 小于最大长度的句子用PAD补足
        batch_y = [x + ([self.voca.PAD] * (max_ly - len(x))) for x in batch_y]

        assert 0 not in length_x
        assert 0 not in length_y

        return batch_x, length_x, batch_y, length_y

    def _cell(self, keep_prob):
        cell = rnn.LSTMCell(num_units=self.rnn_hidden_size)
        return rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)

    def _encoder(self, keep_prob, x_embedding, x_sequence_length, batch_size):
        num_layers = self.lstm_dims
        cell_f = rnn.MultiRNNCell([self._cell(keep_prob) for _ in range(num_layers // 2)])
        cell_b = rnn.MultiRNNCell([self._cell(keep_prob) for _ in range(num_layers // 2)])
        # 计算encoder
        output, states = tf.nn.bidirectional_dynamic_rnn(cell_bw=cell_b, cell_fw=cell_f, inputs=x_embedding,
                                                         initial_state_bw=cell_b.zero_state(batch_size, tf.float32),
                                                         initial_state_fw=cell_f.zero_state(batch_size, tf.float32),
                                                         sequence_length=x_sequence_length)
        encoder_outputs = tf.concat(output, 2)
        encoder_state = []
        for layer_id in range(num_layers // 2):
            encoder_state.append(states[0][layer_id])  # forward
            encoder_state.append(states[1][layer_id])  # backward
        encoder_state = tuple(encoder_state)
        return encoder_outputs, encoder_state

    def _decoder(self, keep_prob, encoder_output, encoder_state, batch_size, scope, helper, reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            attention_states = encoder_output
            cell = rnn.MultiRNNCell([self._cell(keep_prob) for _ in range(self.lstm_dims)])
            attention_mechanism = seq2seq.BahdanauAttention(self.rnn_hidden_size, attention_states)  # attention
            decoder_cell = seq2seq.AttentionWrapper(cell, attention_mechanism,
                                                    attention_layer_size=self.rnn_hidden_size // 2)
            decoder_cell = rnn.OutputProjectionWrapper(decoder_cell, self.rnn_hidden_size, reuse=reuse,
                                                       activation=tf.nn.leaky_relu)
            decoder_initial_state = decoder_cell.zero_state(batch_size, tf.float32).clone(cell_state=encoder_state)

            output_layer = tf.layers.Dense(self.voca.word_num,
                                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                           activation=tf.nn.leaky_relu)
            decoder = seq2seq.BasicDecoder(decoder_cell, helper, decoder_initial_state, output_layer=output_layer)
            output, _, _ = seq2seq.dynamic_decode(decoder, maximum_iterations=self.max_sentence_length,
                                                  impute_finished=True)

            # tf.summary.histogram('decoder', output)
        return output

    def _model(self):
        graph = tf.Graph()
        with graph.as_default():
            embedding = tf.get_variable('embedding', initializer=self.embed_np)
            # embedding = tf.Variable(embed, trainable=True, name='embedding')  # 词向量
            lr = tf.placeholder(tf.float32, [], name='learning_rate')
            # 输入数据
            x_input = tf.placeholder(tf.int32, [None, None], name='x_input')  # 输入数据X
            x_sequence_length = tf.placeholder(tf.int32, [None], name='x_length')  # 输入数据每一条的长度
            x_embedding = tf.nn.embedding_lookup(embedding, x_input)  # 将输入的one-hot编码转换成向量
            y_input = tf.placeholder(tf.int32, [None, None], name='y_input')  # 输入数据Y
            y_sequence_length = tf.placeholder(tf.int32, [None], name='y_length')  # 每一个Y的长度
            batch_size = tf.placeholder(tf.int32, [], name='batch_size')
            keep_prob = tf.placeholder(tf.float32, [], name='keep_prob')

            encoder_output, encoder_state = self._encoder(keep_prob, x_embedding, x_sequence_length, batch_size)

            left_side = tf.fill([batch_size, 1], self.voca.SOS)
            right_side = tf.strided_slice(y_input, [0, 0], [batch_size, -1], [1, 1])
            preprocessed_targets = tf.concat([left_side, right_side], 1)
            y_embedding = tf.nn.embedding_lookup(embedding, preprocessed_targets)  # 对Y向量化

            training_helper = seq2seq.TrainingHelper(inputs=y_embedding, sequence_length=y_sequence_length)
            predict_helper = seq2seq.GreedyEmbeddingHelper(embedding, tf.fill([batch_size], self.voca.SOS),
                                                           self.voca.EOS)

            train_output = self._decoder(keep_prob, encoder_output, encoder_state, batch_size, 'decode',
                                         training_helper)
            predict_output = self._decoder(keep_prob, encoder_output, encoder_state, batch_size, 'decode',
                                           predict_helper, True)

            # loss function
            training_logits = tf.identity(train_output.rnn_output, name='training_logits')
            predicting_logits = tf.identity(predict_output.sample_id, name='predicted_id')

            # target = tf.slice(y_input, [0, 1], [-1, -1])
            # target = tf.concat([tf.fill([batch_size, 1], self.word2index['GO']), y_input], 1)

            masks = tf.sequence_mask(y_sequence_length, dtype=tf.float32, name='mask')

            loss = seq2seq.sequence_loss(training_logits, y_input, masks)
            optimizer = tf.train.AdamOptimizer(lr)
            gradients = optimizer.compute_gradients(loss)
            capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
            train_op = optimizer.apply_gradients(capped_gradients)
            # predicting_logits = tf.nn.softmax(predicting_logits, axis=1)
            tf.summary.scalar('loss', loss)
            tf.summary.scalar('learning rate', lr)
            # tf.summary.tensor_summary('learning rate', lr)

        return graph, loss, train_op, predicting_logits

    def test_data(self, data):
        batch_size = 32
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
                    objector = np.vectorize(lambda s: self.voca.index2word(s))
                    output = np.argmax(output[0], 1)  # 按行取最大值
                    output = objector(output)
                    target = objector(batch_y[0])
                    target_in = objector(batch_x[0])
                    # predict_output = np.argmax(predict_output[0], 1)  # 按行取最大值
                    predict_output = objector(predict_output[0])
                    print(
                        'input: {}\noutput: {}\ntarget-input: {}\ntarget-output: {}\ntrain-output: {}\npredict-output: {}'.
                            format(x[0], y[0], ' '.join(target_in), ' '.join(target), ' '.join(output),
                                   ' '.join(predict_output)))
                    printed = True

                aver_loss += loss
        return aver_loss / batch_num

    def train(self, batch_size, learning_rate, max_epoch=1):
        learning_rate_set = [0, 1e-10, 1e-9, 1e-9, 1e-9, 1e-8, 1e-7, 1e-5, 1e-4, 1e-3, 1e-3, 1e-3, 1e-2]
        train_data, valid_data, test_data = read_data.read_data_db()
        self.test_data(valid_data)
        _lr = learning_rate
        g = self.graph
        tr_batch_num = train_data.size // batch_size
        print("start training, batch size {}, max epoch: {}, max batch: {}".format(batch_size, max_epoch, tr_batch_num))
        loss_sum = 0
        start_time = time.time()
        with self.sess.as_default():
            for epoch in range(max_epoch):
                for i in range(tr_batch_num):
                    x, y = train_data.next_batch(batch_size)
                    batch_x, length_x, batch_y, length_y = self._preprocess_data(x, y)
                    assert len(batch_x) == len(batch_y) == len(length_x) == len(length_y)

                    feed_dict = {g.get_tensor_by_name('x_input:0'): batch_x,
                                 g.get_tensor_by_name('x_length:0'): length_x,
                                 g.get_tensor_by_name('y_input:0'): batch_y,
                                 g.get_tensor_by_name('y_length:0'): length_y,
                                 g.get_tensor_by_name('learning_rate:0'): _lr,
                                 g.get_tensor_by_name('batch_size:0'): len(length_x),
                                 g.get_tensor_by_name('keep_prob:0'): 0.8
                                 }

                    loss_var, *_ = self.sess.run([self.loss, self.train_op], feed_dict)
                    # self.writer.add_summary(me)
                    loss_sum += loss_var
                    if (i + 1) % 100 == 0:
                        during_time = time.time() - start_time
                        start_time = time.time()
                        print("training epoch {}/{}, learning rate = {}, batch {}/{}, loss={:.6f}, during {:.1f} sec"
                              .format(epoch + 1, max_epoch, _lr, i + 1, tr_batch_num, loss_sum / 100, during_time))
                        loss_sum = 0
                        with open('learning_rate', encoding='utf-8') as f:
                            _lr = float(f.read())
                        # _lr = learning_rate_set[int(loss_var)]

                    if (i + 1) % 1000 == 0:
                        loss_var = self.test_data(valid_data)
                        print("valid result: loss={0}".format(loss_var))
                        self.save_model()
                        start_time = time.time()
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
        question = [question + ['EOS']]
        question = np.array(question)
        question = np.vectorize(lambda x: self.voca.word2index(x))(question)
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
        answer = np.vectorize(lambda x: self.voca.index2word(x))(answer)
        print(answer)


if __name__ == '__main__':
    a = seq2seqModel()
    i = 3
    a.train(batch_size=64, learning_rate=0.0005, max_epoch=1)

# reference https://github.com/AbrahamSanders/seq2seq-chatbot/
