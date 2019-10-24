import time
import numpy as np
import tensorflow as tf
from sklearn import metrics
import utils


class Model:
    def __init__(self, size_layer, num_layers, embedded_size, dict_size, dimension_output, learning_rate,
                 attention_size):
        self.size_layer = size_layer
        self.num_layers = num_layers
        self.embedded_size = embedded_size
        self.dict_size = dict_size
        self.dimension_output = dimension_output
        self.learning_rate = learning_rate
        self.attention_size = attention_size

    @staticmethod
    def cells(size_layer, reuse=False):
        return tf.nn.rnn_cell.LSTMCell(size_layer, initializer=tf.orthogonal_initializer(), reuse=reuse)

    def main(self):
        # (B, S) batch_size/seq_len
        self.X = tf.placeholder(tf.int32, [None, None])
        # (?, 2)
        self.Y = tf.placeholder(tf.float32, [None, self.dimension_output])
        # (20336, 128)
        encoder_embeddings = tf.Variable(tf.random_uniform([self.dict_size, self.embedded_size], -1, 1))
        # (B, S, 128)
        encoder_embedded = tf.nn.embedding_lookup(encoder_embeddings, self.X)
        # 2层LSTM 每层128个神经元
        rnn_cells = tf.nn.rnn_cell.MultiRNNCell(
            [Model.cells(size_layer=self.size_layer) for _ in range(self.num_layers)])
        # outputs: (B, S, 128) last_state: c/h (B, 128)*2
        outputs, last_state = tf.nn.dynamic_rnn(rnn_cells, encoder_embedded, dtype=tf.float32)
        # (150,)
        attention_w = tf.get_variable("attention_v", [self.attention_size], tf.float32)
        # last_state[-1].h: (B, 128)    tf.expand_dims(last_state[-1].h, 1): (B, 1, 128)    query:  (B, 1, 150)
        query = tf.layers.dense(tf.expand_dims(last_state[-1].h, 1), self.attention_size)
        # (B, S, 150)
        keys = tf.layers.dense(outputs, self.attention_size)
        # keys + query: (B, S, 150)     tf.tanh(keys + query): (B, S, 150)     attention_w * tf.tanh(keys + query):
        # (B, S, 150)  align: (B, S)
        align = tf.reduce_sum(attention_w * tf.tanh(keys + query), [2])
        # (B, S)
        align = tf.nn.softmax(align)
        # tf.transpose(outputs, [0, 2, 1]): (?, 128, ?)     tf.expand_dims(align, 2): (?, ?, 1)
        # tf.matmul(tf.transpose(outputs, [0, 2, 1]), tf.expand_dims(align, 2)): (?, 128, 1)
        # outputs: (?, 128)
        outputs = tf.squeeze(tf.matmul(tf.transpose(outputs, [0, 2, 1]), tf.expand_dims(align, 2)), 2)
        # size_layer: 128   dimension_output: 2
        W = tf.get_variable('w', shape=(self.size_layer, self.dimension_output),
                            initializer=tf.orthogonal_initializer())
        # dimension_output: 2
        b = tf.get_variable('b', shape=(self.dimension_output), initializer=tf.zeros_initializer())
        # outputs: (?, 128)     W: (128, 2)     b: (2,)
        self.logits = tf.matmul(outputs, W) + b
        # Y: (?, 2)
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y))
        # 优化器
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
        # 准确与否
        correct_pred = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
        # 计算准确率
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


class LstmRnnAttention(object):
    def __init__(self):
        self.size_layer = 128
        self.num_layers = 2
        self.embedded_size = 128
        self.learning_rate = 1e-3
        self.maxlen = 50
        self.batch_size = 128
        self.attention_size = 150
        self.vocabulary_size = None
        self.dimension_output = None

    def main(self):
        trainset, train_X, test_X, train_Y, test_Y, train_onehot, test_onehot = utils.utils.get_train_test()
        self.dimension_output = len(trainset.target_names)

        data, count, dictionary, rev_dictionary, vocabulary_size = utils.utils.build_dataset(
            ' '.join(trainset.data).split())
        self.vocabulary_size = vocabulary_size

        GO = dictionary['GO']
        PAD = dictionary['PAD']
        EOS = dictionary['EOS']
        UNK = dictionary['UNK']

        tf.reset_default_graph()
        sess = tf.InteractiveSession()
        model = Model(size_layer=self.size_layer, num_layers=self.num_layers, embedded_size=self.embedded_size,
                      dict_size=self.vocabulary_size + 4, dimension_output=self.dimension_output,
                      learning_rate=self.learning_rate, attention_size=self.attention_size)
        sess.run(tf.global_variables_initializer())

        EARLY_STOPPING, CURRENT_CHECKPOINT, CURRENT_ACC, EPOCH = 5, 0, 0, 0
        while True:
            lasttime = time.time()
            if CURRENT_CHECKPOINT == EARLY_STOPPING:
                print('break epoch:%d\n' % (EPOCH))
                break

            train_acc, train_loss, test_acc, test_loss = 0, 0, 0, 0
            for i in range(0, (len(train_X) // self.batch_size) * self.batch_size, self.batch_size):
                batch_x = utils.utils.str_idx(train_X[i:i + self.batch_size], dictionary, self.maxlen)
                acc, loss, _ = sess.run([model.accuracy, model.cost, model.optimizer],
                                        feed_dict={model.X: batch_x, model.Y: train_onehot[i:i + self.batch_size]})
                train_loss += loss
                train_acc += acc

            for i in range(0, (len(test_X) // self.batch_size) * self.batch_size, self.batch_size):
                batch_x = utils.utils.str_idx(test_X[i:i + self.batch_size], dictionary, self.maxlen)
                acc, loss = sess.run([model.accuracy, model.cost],
                                     feed_dict={model.X: batch_x, model.Y: test_onehot[i:i + self.batch_size]})
                test_loss += loss
                test_acc += acc

            train_loss /= (len(train_X) // self.batch_size)
            train_acc /= (len(train_X) // self.batch_size)
            test_loss /= (len(test_X) // self.batch_size)
            test_acc /= (len(test_X) // self.batch_size)

            if test_acc > CURRENT_ACC:
                print('epoch: %d, pass acc: %f, current acc: %f' % (EPOCH, CURRENT_ACC, test_acc))
                CURRENT_ACC = test_acc
                CURRENT_CHECKPOINT = 0
            else:
                CURRENT_CHECKPOINT += 1

            print('time taken:', time.time() - lasttime)
            print(
                'epoch: %d, training loss: %f, training acc: %f, valid loss: %f, valid acc: %f\n' % (EPOCH, train_loss,
                                                                                                     train_acc,
                                                                                                     test_loss,
                                                                                                     test_acc))
            EPOCH += 1

        logits = sess.run(model.logits, feed_dict={model.X: utils.utils.str_idx(test_X, dictionary, self.maxlen)})
        print(metrics.classification_report(test_Y, np.argmax(logits, 1), target_names=trainset.target_names))



