import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
import math


class bilstm(object):
    def __init__(self, parameters, **kwargs):
        print 'Building model ...'
        self.params = parameters
        # global
        # self.batch_size = self.params['batch_size']
        self.tag_size = self.params['tag_size']
        self.char_tag_size = self.params['char_tag_size']
        self.clip_norm = self.params['clip_norm']
        # word
        self.vocab_size = self.params['vocab_size']
        self.word_input_dim = self.params['word_input_dim']
        self.word_hidden_dim = self.params['word_hidden_dim']
        # self.word_bidirect = self.params['word_bidirect']
        # character
        self.char_vocab_size = self.params['char_vocab_size']
        self.char_input_dim = self.params['char_input_dim']
        self.char_hidden_dim = self.params['char_hidden_dim']
        # self.char_bidirect = self.params['char_bidirect']
        self.total_loss = 0

    def _word_embedding(self, word_input_ids):
        with tf.variable_scope('word_embedding') as vs:
            if self.params['use_word2vec']:
                W_word = tf.get_variable('Word_embedding',
                                          initializer=self.params['embedding_initializer'],
                                          trainable=self.params['fine_tune_w2v'],
                                          dtype=tf.float32)
            else:
                W_word = tf.Variable(tf.random_uniform([self.vocab_size, self.word_input_dim], -0.25, 0.25),
                                          trainable=True,
                                          name='Word_embedding',
                                          dtype=tf.float32)

            embedded_words = tf.nn.embedding_lookup(W_word, word_input_ids,name='embedded_words')
            print vs.name, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name)

        return embedded_words

    def _char_embedding(self, char_input_ids):
        with tf.variable_scope('char_embedding') as vs:
            drange = np.sqrt(6. / (np.sum([self.char_vocab_size-1, self.char_input_dim])))
            char_initializer = tf.concat([tf.zeros([1, self.char_input_dim]),
                                         tf.random_uniform([self.char_vocab_size-1, self.char_input_dim], -0.25, 0.25)],
                                         axis=0)
            W_char = tf.Variable(char_initializer,
                                 trainable=True,
                                 name='Char_embedding',
                                 dtype=tf.float32)

            # (batch_size, max_sent_len, max_char_len, char_input_dim)
            embedded_chars = tf.nn.embedding_lookup(W_char, char_input_ids, name='embedded_chars')
            print vs.name, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name)

        return embedded_chars

    def _char_lstm_word(self, embedded_chars, word_lengths):
        with tf.variable_scope('char_lstm') as vs:
            s = tf.shape(embedded_chars)
            new_lstm_embedded_chars = tf.reshape(embedded_chars, shape=[s[0]*s[1], s[2], self.char_input_dim])
            # (batch_size*max_sent_len, max_char_len, char_input_dim)
            real_word_lengths = tf.reshape(word_lengths, shape=[s[0]*s[1]])
            if self.params['num_layers'] > 1:
                fw_cells = []
                bw_cells = []
                for _ in range(self.params['num_layers']):
                    fw_cell = rnn.LSTMCell(self.char_hidden_dim, state_is_tuple=True)
                    fw_cell = rnn.DropoutWrapper(fw_cell, output_keep_prob=self.dropout_keep_prob)
                    fw_cells.append(fw_cell)
                    bw_cell = rnn.LSTMCell(self.char_hidden_dim, state_is_tuple=True)
                    bw_cell = rnn.DropoutWrapper(bw_cell, output_keep_prob=self.dropout_keep_prob)
                    bw_cells.append(bw_cell)
                char_fw_cell = rnn.MultiRNNCell(fw_cells, state_is_tuple=True)
                char_bw_cell = rnn.MultiRNNCell(bw_cells, state_is_tuple=True)
            else:
                char_fw_cell = rnn.LSTMCell(self.char_hidden_dim, state_is_tuple=True)
                char_bw_cell = rnn.LSTMCell(self.char_hidden_dim, state_is_tuple=True)

            (seq_fw, seq_bw), (fw_state_tuples, bw_state_tuples) = \
                tf.nn.bidirectional_dynamic_rnn(char_fw_cell,
                                                char_bw_cell,
                                                new_lstm_embedded_chars,
                                                sequence_length=real_word_lengths,
                                                dtype=tf.float32,
                                                swap_memory=True)

            if self.params['num_layers'] > 1:
                char_fw_final_out = fw_state_tuples[-1][1]
                char_bw_final_out = bw_state_tuples[-1][1]
            else:
                char_fw_final_out = fw_state_tuples[1]
                char_bw_final_out = bw_state_tuples[1]
            # print char_fw_final_out.get_shape()
            # print char_bw_final_out.get_shape()
            char_output = tf.concat([char_fw_final_out, char_bw_final_out], axis=-1, name='Char_BiLSTM')
            char_output = tf.reshape(char_output, shape=[s[0], s[1], 2*self.char_hidden_dim]) # (batch_size, max_sent, 2*char_hidden)
            char_hiddens = tf.concat([seq_fw, seq_bw], axis=-1, name='char_hidden_sequence')
            char_hiddens = tf.reshape(char_hiddens, shape=[s[0], s[1], s[2], 2*self.char_input_dim]) # (batch_size*max_sent, max_char, 2*char_hidden)
            print vs.name, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name)

        return char_output, char_hiddens

    def _char_cnn(self, embedded_chars):
        with tf.variable_scope('char_cnn') as vs:
            s = tf.shape(embedded_chars)
            new_cnn_embedded_chars = tf.reshape(embedded_chars, shape=[s[0]*s[1], s[2], self.char_input_dim])
            # (batch_size*max_sent_len, max_char_len, char_input_dim)

            filter_shape = [self.params['filter_size'], self.char_input_dim, self.params['num_filters']]
            W_filter = tf.get_variable("W_filter",
                                       shape=filter_shape,
                                       initializer=tf.contrib.layers.xavier_initializer())
            b_filter = tf.Variable(tf.constant(0.0, shape=[self.params['num_filters']]), name='f_filter')

            # input: [batch, in_width, in_channels]
            # filter: [filter_width, in_channels, out_channels]
            conv = tf.nn.conv1d(new_cnn_embedded_chars,
                                W_filter,
                                stride=1,
                                padding="SAME",
                                name='conv1')
            # (batch_size*max_sent_len, out_width, num_filters)
            # print 'conv: ', conv.get_shape()

            # h_conv1 = tf.nn.relu(tf.nn.bias_add(conv, b_filter, name='add bias'))
            h_conv1 = tf.nn.relu(conv + b_filter)
            h_expand = tf.expand_dims(h_conv1, -1)
            # print 'h_expand: ', h_expand.get_shape()
            # (batch_size*max_sent_len, out_width, num_filters, 1)

            h_pooled = tf.nn.max_pool(h_expand,
                                      ksize=[1, self.params['max_char_len'], 1, 1],
                                      strides=[1, self.params['max_char_len'], 1, 1],
                                      padding="SAME",
                                      name='pooled')
            # print 'pooled: ', h_pooled.get_shape()
            # (batch_size*max_sent_len, num_filters, 1)

            char_pool_flat = tf.reshape(h_pooled, [s[0], s[1], self.params['num_filters']])
            # (batch_size, max_sent, num_filters)

            print vs.name, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name)

        return char_pool_flat

    def _word_lstm(self, embedded_words, sequence_lengths):
        with tf.variable_scope('word_bilstm') as vs:
            if self.params['num_layers'] > 1:
                fw_cells = []
                bw_cells = []
                for _ in range(self.params['num_layers']):
                    fw_cell = rnn.LSTMCell(self.word_hidden_dim, state_is_tuple=True)
                    fw_cell = rnn.DropoutWrapper(fw_cell, output_keep_prob=self.dropout_keep_prob)
                    fw_cells.append(fw_cell)
                    bw_cell = rnn.LSTMCell(self.word_hidden_dim, state_is_tuple=True)
                    bw_cell = rnn.DropoutWrapper(bw_cell, output_keep_prob=self.dropout_keep_prob)
                    bw_cells.append(bw_cell)
                word_fw_cell = rnn.MultiRNNCell(fw_cells, state_is_tuple=True)
                word_bw_cell = rnn.MultiRNNCell(bw_cells, state_is_tuple=True)
            else:
                word_fw_cell = rnn.LSTMCell(self.word_hidden_dim, state_is_tuple=True)
                word_bw_cell = rnn.LSTMCell(self.word_hidden_dim, state_is_tuple=True)
                #if self.params['dropout']:
                #   word_fw_cell = rnn.DropoutWrapper(word_fw_cell, output_keep_prob=self.dropout_keep_prob)
                #   word_bw_cell = rnn.DropoutWrapper(word_bw_cell, output_keep_prob=self.dropout_keep_prob)

            (output_seq_fw, output_seq_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                                                word_fw_cell,
                                                word_bw_cell,
                                                embedded_words,
                                                sequence_length=sequence_lengths,
                                                dtype=tf.float32,
                                                swap_memory=True)


            # (batch_size, max_sent_len, 2*word_hidden_dim)
            word_biLSTM_output = tf.concat([output_seq_fw, output_seq_bw], axis=-1, name='BiLSTM')
            #if self.params['dropout']:
            #    word_biLSTM_output = tf.nn.dropout(word_biLSTM_output, self.dropout_keep_prob)
            print vs.name, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name)

        return word_biLSTM_output, output_seq_fw, output_seq_bw

    def _label_prediction(self, word_bilstm_output):
        with tf.variable_scope('output_layers') as vs:
            # (batch_size*max_sent_len, 2*word_hidden_dim)
            reshape_biLSTM_output = tf.reshape(word_bilstm_output, [-1, 2*self.word_hidden_dim])

            W_fc1 = tf.get_variable("softmax_W_fc1",
                                    shape=[2*self.word_hidden_dim, self.word_hidden_dim],
                                    initializer=tf.contrib.layers.xavier_initializer())
            b_fc1 = tf.Variable(tf.constant(0.0, shape=[self.word_hidden_dim]), name='softmax_b_fc1')
            o_fc1 = tf.nn.relu(tf.nn.xw_plus_b(reshape_biLSTM_output, W_fc1, b_fc1))

            W_out = tf.get_variable("softmax_W_out",
                                    shape=[self.word_hidden_dim, self.tag_size],
                                    initializer=tf.contrib.layers.xavier_initializer())
            b_out = tf.Variable(tf.constant(0.0, shape=[self.tag_size]), name='softmax_b_out')
            # self.predictions = tf.matmul(self.biLSTM_output, W_out) + b_out
            predictions = tf.nn.xw_plus_b(o_fc1, W_out, b_out, name='softmax_output')
            # (batch_size, max_sent_len, tag_size)
            logits = tf.reshape(predictions, [self.batch_size, self.max_sent_len, self.tag_size], name='logits')

            print vs.name, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name)

        return logits

    def _loss_cal(self, logits):
        with tf.variable_scope('loss') as vs:
            if self.params['use_crf_loss']:
                log_likelihood, self.transition_params = \
                    tf.contrib.crf.crf_log_likelihood(logits,
                                                      self.tag_input_ids,
                                                      self.sequence_lengths)
                word_loss = tf.reduce_mean(-log_likelihood, name='crf_negloglik_loss')
                # print self.transition_params.name
            else:
                # add softmax loss
                self.pred_tags = tf.cast(tf.argmax(logits, axis=-1), tf.int32)
                losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                        labels=self.tag_input_ids)
                mask = tf.sequence_mask(self.sequence_lengths)
                losses = tf.boolean_mask(losses, mask)
                word_loss = tf.reduce_mean(losses)

            print vs.name, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name)
        return word_loss

    def _word_lm(self, word_seq_fw, word_seq_bw):
        lm_loss = 0
        with tf.variable_scope('forward_lm') as vs:
            W_forward = tf.get_variable('softmax_for_W',
                                          shape=[self.word_hidden_dim, self.params['lm_vocab_size']],
                                          initializer=tf.contrib.layers.xavier_initializer())
            b_forward = tf.Variable(tf.constant(0.0, shape=[self.params['lm_vocab_size']]), name='softmax_for_b')
            bilstm_for_reshape = tf.reshape(word_seq_fw, shape=[-1, self.word_hidden_dim])
            pred_forward = tf.nn.xw_plus_b(bilstm_for_reshape, W_forward, b_forward, name='softmax_forward')
            logits_forward = tf.reshape(pred_forward,
                                        shape=[self.batch_size, self.max_sent_len, self.params['lm_vocab_size']],
                                        name='logits_forward')
            for_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_forward,
                                                                        labels=self.forward_words)
            forward_lm_loss = tf.reduce_mean(tf.boolean_mask(for_losses, tf.sequence_mask(self.sequence_lengths)))
            lm_loss += forward_lm_loss

            print vs.name, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name)

        with tf.variable_scope('backward_lm') as vs:
            W_backward = tf.get_variable('softmax_bak_W',
                                          shape=[self.word_hidden_dim, self.params['lm_vocab_size']],
                                          initializer=tf.contrib.layers.xavier_initializer())
            b_backward = tf.Variable(tf.constant(0.0, shape=[self.params['lm_vocab_size']]), name='softmax_bak_b')
            bilstm_bak_reshape = tf.reshape(word_seq_bw, shape=[-1, self.word_hidden_dim])
            pred_backward = tf.nn.xw_plus_b(bilstm_bak_reshape, W_backward, b_backward, name='softmax_backward')
            logits_backward = tf.reshape(pred_backward,
                                         shape=[self.batch_size, self.max_sent_len, self.params['lm_vocab_size']],
                                         name='logits_backward')
            bak_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_backward,
                                                                        labels=self.backward_words)
            backward_lm_loss = tf.reduce_mean(tf.boolean_mask(bak_losses, tf.sequence_mask(self.sequence_lengths)))
            lm_loss += backward_lm_loss

            print vs.name, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name)

        return lm_loss

    def build(self):
        # place holders
        self.word_input_ids = tf.placeholder(tf.int32, [None, None], name='word_input')
        self.tag_input_ids = tf.placeholder(tf.int32, [None, None], name='tag_input')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        self.sequence_lengths = tf.placeholder(tf.int32, [None], name='sequence_lengths')
        self.char_input_ids = tf.placeholder(tf.int32, [None, None, None], name='char_input')
        self.word_lengths = tf.placeholder(tf.int32, [None, None], name='word_lengths')

        # dynamic number
        self.batch_size = tf.shape(self.word_input_ids)[0]
        self.max_sent_len = tf.shape(self.word_input_ids)[1]
        self.max_char_len = tf.shape(self.char_input_ids)[2]

        # word embedding
        embedded_words = self._word_embedding(self.word_input_ids)
        if self.params['dropout']:
            embedded_words = tf.nn.dropout(embedded_words, self.dropout_keep_prob)

        if self.params['char_encode'] == 'lstm':

            # char embedding
            embedded_chars = self._char_embedding(self.char_input_ids)
            if self.params['dropout']:
                embedded_chars = tf.nn.dropout(embedded_chars, self.dropout_keep_prob)
            # char encoding
            char_output, char_hiddens = self._char_lstm_word(embedded_chars, self.word_lengths)
            word_lstm_input = tf.concat([embedded_words, char_output], axis=-1)

        elif self.params['char_encode'] == 'cnn':

            # char embedding
            embedded_chars = self._char_embedding(self.char_input_ids)
            if self.params['dropout']:
                embedded_chars = tf.nn.dropout(embedded_chars, self.dropout_keep_prob)
            # char encoding
            char_output = self._char_cnn(embedded_chars)
            word_lstm_input = tf.concat([embedded_words, char_output], axis=-1)

        else:
            word_lstm_input = embedded_words

        if self.params['dropout']:
            word_lstm_input = tf.nn.dropout(word_lstm_input, self.dropout_keep_prob)

        # word encoding
        word_bilstm_output, word_seq_fw, word_seq_bw = self._word_lstm(word_lstm_input, self.sequence_lengths)

        if self.params['dropout']:
            word_bilstm_output = tf.nn.dropout(word_bilstm_output, self.dropout_keep_prob)

        # intermediate fc layers
        self.logits = self._label_prediction(word_bilstm_output)
        # calculate loss
        self.word_loss = self._loss_cal(self.logits)
        self.total_loss += self.word_loss

        if self.params['word_lm']:
            self.forward_words = tf.placeholder(tf.int32, [None, None], name='forward_words')
            self.backward_words = tf.placeholder(tf.int32, [None, None], name='backward_words')
            self.word_lm_loss = self._word_lm(word_seq_fw, word_seq_bw)
            self.total_loss += self.word_lm_loss

        #Variable for Learning rate decay
        global_step = tf.Variable(0, trainable=False)

        # optimization
        if self.params['lr_method'].lower() == 'adam':
            optimizer_total = tf.train.AdamOptimizer(self.params['lr_rate'])
        elif self.params['lr_method'].lower() == 'adagrad':
            optimizer_total = tf.train.AdagradOptimizer(self.params['lr_rate'])
        elif self.params['lr_method'].lower() == 'adadelta':
            optimizer_total = tf.train.AdadeltaOptimizer(self.params['lr_rate'])
        elif self.params['lr_method'].lower() == 'sgd':
            if self.params['weight_decay']>0:
                learning_rate = tf.train.exponential_decay(self.params['lr_rate'],global_step, self.params['decay_steps'],
                                                           self.params['weight_decay'])
            else:
                learning_rate = self.params['lr_rate']
            optimizer_total = tf.train.GradientDescentOptimizer(learning_rate)
        elif self.params['lr_method'].lower() == 'rmsprop':
            optimizer_total = tf.train.RMSPropOptimizer(self.params['lr_rate'])
        elif self.params['lr_method'].lower() == 'momentum':
            if self.params['weight_decay']>0:
                learning_rate = tf.train.exponential_decay(self.params['lr_rate'],global_step, self.params['decay_steps'],
                                                           self.params['weight_decay'])
            else:
                learning_rate = self.params['lr_rate']
            optimizer_total = tf.train.MomentumOptimizer(learning_rate, self.params['momentum'])

        if self.params['clip_norm'] > 0:
            grads, vs = zip(*optimizer_total.compute_gradients(self.total_loss))
            grads, gnorm = tf.clip_by_global_norm(grads, self.params['clip_norm'])
            self.total_train_op = optimizer_total.apply_gradients(zip(grads, vs),global_step=global_step)
        else:
            self.total_train_op = optimizer_total.minimize(self.total_loss,global_step=global_step)

        return
