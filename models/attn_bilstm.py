#!/usr/bin/python
# -*- coding:utf8 -*-

"""
    @author:xiaotian zhao
    @time:3/20/19
"""

import pickle
import tensorflow as tf


class AttnBiLSTM(object):
    def __init__(self, hps):
        self.hps = hps
        self.embed_size = self.hps.embed_size
        self.vocab_size = self.hps.vocab_size
        self.max_seq_len = self.hps.max_seq_len
        self.use_embed = self.hps.use_embed
        self.zero_pad = self.hps.zero_pad
        self.rnn_hidden_size = self.hps.rnn_hidden_size
        self.num_class = self.hps.num_class
        self.cell = self.hps.cell
        self.lr = self.hps.lr
        self.eval = False
        self.l2_reg_lambda = 1e-5

        self.cuda_available = tf.test.is_built_with_cuda() and self.hps.use_cuda
        self.device_name = '/gpu:0' if self.cuda_available else '/cpu:0'
        self.trunc_norm_init = tf.truncated_normal_initializer(stddev=0.1)

    def _make_feeddict(self, x_batch, y_batch, len_batch, keep_dropouts):
        feed_dict = {}
        feed_dict[self.x_batch] = x_batch
        if y_batch:
            feed_dict[self.y_batch] = y_batch
        if len_batch:
            feed_dict[self.len_batch] = len_batch

        feed_dict[self.keep_dropouts] = keep_dropouts
        return feed_dict

    def _add_placeholders(self):
        with tf.name_scope("inputs"):
            self.x_batch = tf.placeholder(tf.int32, [None, None])
            self.len_batch = tf.placeholder(tf.int32, [None])
            self.y_batch = tf.placeholder(tf.int64, [None])
            self.keep_dropouts = tf.placeholder(tf.float32, [3])

    def _add_embed_layer(self):
        with tf.name_scope("embed_layer"):
            if self.use_embed:
                w2v_weights = pickle.load(open(self.hps.pretrain_embed_file, 'rb'))
                self.w2v_matrix = tf.Variable(w2v_weights, name='w2v_matrix')
            else:
                self.w2v_matrix = tf.get_variable(
                    "w2v_matrix",
                    [self.vocab_size, self.embed_size],
                    dtype=tf.float32,
                    initializer=self.trunc_norm_init
                )

            if self.zero_pad:
                self.w2v_matrix = tf.concat(
                    (tf.zeros(shape=[1, self.embed_size]), self.w2v_matrix[1:, :]),
                    0
                )

            self.word_embeds = tf.nn.embedding_lookup(self.w2v_matrix, self.x_batch)
            if not self.eval:
                self.word_embeds = tf.nn.dropout(self.word_embeds, keep_prob=self.keep_dropouts[0])

    def _add_lstm_layer(self):
        with tf.name_scope("lstm_layer"):
            if self.cell.lower() == 'lstm':
                fw_cell = tf.contrib.rnn.BasicLSTMCell(self.rnn_hidden_size)
                bw_cell = tf.contrib.rnn.BasicLSTMCell(self.rnn_hidden_size)
            elif self.cell.lower() == 'gru':
                fw_cell = tf.contrib.rnn.GRUCell(self.rnn_hidden_size)
                bw_cell = tf.contrib.rnn.GRUCell(self.rnn_hidden_size)
            else:
                raise NotImplementedError("only support gru/lstm, check your params \'cell\'")

        self.rnn_outputs, self.final_states_tuple = tf.nn.bidirectional_dynamic_rnn(
            tf.nn.rnn_cell.DropoutWrapper(fw_cell, output_keep_prob=self.keep_dropouts[1]),
            tf.nn.rnn_cell.DropoutWrapper(bw_cell, output_keep_prob=self.keep_dropouts[1]),
            self.word_embeds,
            sequence_length=self.len_batch,
            dtype=tf.float32
        )
        self.rnn_outputs = tf.add(self.rnn_outputs[0], self.rnn_outputs[1])

    def _add_attention_layer(self):
        with tf.name_scope("attn_layer"):
            self.M = tf.nn.tanh(self.rnn_outputs)
            self.attn_w = tf.get_variable(
                "attn_w",
                [self.rnn_hidden_size, 1],
                dtype=tf.float32,
                initializer=self.trunc_norm_init
            )

            self.alpha = tf.map_fn(
                lambda x: tf.matmul(x, self.attn_w),
                self.M
            )
            self.alpha = tf.transpose(self.alpha, [0, 2, 1])

            self.context = tf.nn.tanh(tf.squeeze(tf.matmul(self.alpha, self.M), axis=1))

    def _add_output_layer(self):
        with tf.name_scope("output_layer"):
            self.output_w = tf.get_variable(
                "output_w",
                [self.rnn_hidden_size, self.num_class],
                dtype=tf.float32,
                initializer=self.trunc_norm_init
            )
            self.output_b = tf.Variable(tf.constant(
                0.1,
                tf.float32,
                [self.num_class]
            ), name='output_b')

            self.distributions = tf.matmul(tf.nn.dropout(self.context, keep_prob=self.keep_dropouts[2]), self.output_w) + self.output_b
            self.predictions = tf.argmax(self.distributions, axis=-1)
        with tf.name_scope("losses"):
            self.l2_loss = 0.0
            # self.l2_loss += tf.nn.l2_loss(self.attn_w)
            # self.l2_loss += tf.nn.l2_loss(self.output_w)
            self.l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])

            self.ce_loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=self.y_batch,
                    logits=self.distributions
                )
            )

            self.loss = self.ce_loss + self.l2_reg_lambda * self.l2_loss
            tf.summary.scalar("loss", self.loss)
            tf.summary.scalar("ce_loss", self.ce_loss)
            tf.summary.scalar("l2_loss", self.l2_loss)

    def _add_train_op(self):
        with tf.name_scope("train_op"):
            if self.hps.optimizer.lower() == 'adam':
                self.optimizer = tf.train.AdamOptimizer(self.lr)
            else:
                self.optimizer = tf.train.AdadeltaOptimizer(self.lr)
            self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)

    def build_graph(self):
        with tf.device(self.device_name):
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self._add_placeholders()
            self._add_embed_layer()
            self._add_lstm_layer()
            self._add_attention_layer()
            self._add_output_layer()
            self._add_train_op()
            self.summaries = tf.summary.merge_all()

    def run_train_step(self, sess, x_batch, y_batch, len_batch, keep_dropouts):
        feed_dict = self._make_feeddict(x_batch, y_batch, len_batch, keep_dropouts)
        to_return = {
            'train_op': self.train_op,
            'loss': self.loss,
            'summaries': self.summaries,
            'global_step': self.global_step,
        }

        return sess.run(to_return, feed_dict)

    def run_eval_step(self, sess, x_batch, y_batch, len_batch, keep_dropouts):
        feed_dict = self._make_feeddict(x_batch, y_batch, len_batch, keep_dropouts)
        to_return = {
            'loss': self.loss,
            'summaries': self.summaries,
            'global_step': self.global_step,
            'predictions': self.predictions
        }

        return sess.run(to_return, feed_dict)