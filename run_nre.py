#!/usr/bin/python
# -*- coding:utf8 -*-

"""
    @author:xiaotian zhao
    @time:10/28/18
"""

import os
import time
import json
import random
import numpy as np
import tensorflow as tf

from NRE.utils.vocab import Vocab
from NRE.utils.data_utils import minibatches, pad_sequences, _pad_sequences, SemEval10task8Dataset, calc_metrics
from NRE.models.attn_bilstm import AttnBiLSTM

from collections import namedtuple

PAD_TOKEN = '<PAD>' # This has a vocab id, which is used to pad the encoder input, decoder input and target sequence
UNKNOWN_TOKEN = '<UNK>' # This has a vocab id, which is used to represent out-of-vocabulary words
START_DECODING = '<SOS>' # This has a vocab id, which is used at the start of every decoder input sequence
STOP_DECODING = '<EOS>' # This has a vocab id, which is used at the end of untruncated target sequences

tf.app.flags.DEFINE_string('train_file', './data/train.txt', '')
tf.app.flags.DEFINE_string('dev_file', './data/valid.txt', '')
tf.app.flags.DEFINE_string('test_file', './data/test.txt', '')
tf.app.flags.DEFINE_string('word_vocab_file', './data/word_vocab', '')
tf.app.flags.DEFINE_string('tag_vocab_file', './data/tag_vocab', '')
tf.app.flags.DEFINE_string('pretrain_embed_file', './data/w2v_matrix', '')

# Important settings
tf.app.flags.DEFINE_string('cell', 'gru', 'must be one of lstm/gru')
tf.app.flags.DEFINE_string('mode', 'all', 'must be one of train/eval/test/all')
tf.app.flags.DEFINE_string('model', 'attn_lstm', 'must be one of text_cnn/text_cnn/text_rnn_and_cnn')
tf.app.flags.DEFINE_boolean('zero_pad', True, 'if True, w2v matrix has zero at row_1')
tf.app.flags.DEFINE_boolean('restore_best_model', False, "if True, restore the best model")
tf.app.flags.DEFINE_boolean('use_embed', True, "if True, use the pretrain embeddings")
tf.app.flags.DEFINE_string('gpu', '0', 'which gpu you want to use')
tf.app.flags.DEFINE_float('gpu_fraction', 0.5, 'gpu fraction')
tf.app.flags.DEFINE_boolean('shuffle', True, 'shuffle the data when generate batches')

# Where to save output
tf.app.flags.DEFINE_string('log_root', 'log', 'Root directory for all logging')
tf.app.flags.DEFINE_string('output_pred', 'output_pred.txt', 'Output Prediction file path')
tf.app.flags.DEFINE_string('output_gold', 'output_gold.txt', 'Output Gold file path')
tf.app.flags.DEFINE_string('script', '/home/xtzhao/Projects/Python/NRE/data/semeval2010_task8_scorer-v1.2.pl', 'Output Gold file path')
tf.app.flags.DEFINE_string('exp_name', 'bi-lstm', 'Name for experiment')
tf.app.flags.DEFINE_string('optimizer', 'adadelta', 'sgd/adam/adadelta')

# Training settings
tf.app.flags.DEFINE_integer('print_every', 20, 'number of epoch')
tf.app.flags.DEFINE_integer('save_every', 20, 'number of epoch')

# Hyperparameters
tf.app.flags.DEFINE_integer('embed_size', 100, 'dimension of word embedding')
tf.app.flags.DEFINE_integer('vocab_size', 0, 'size of word vocab')
tf.app.flags.DEFINE_integer('max_seq_len', 26, 'max sequence length')
tf.app.flags.DEFINE_integer('epoch_num', 2000, 'number of epoch')
tf.app.flags.DEFINE_integer('batch_size', 10, 'batch size')
tf.app.flags.DEFINE_integer('lucky_num', 666, 'random seed number')
tf.app.flags.DEFINE_float('lr', 1.0, 'learning rate')
tf.app.flags.DEFINE_float('adagrad_init_acc', 0.1, 'initial accumulator value for Adagrad')
tf.app.flags.DEFINE_float('l2_reg_lambda', 1e-5, 'l2 regularize lambda')
tf.app.flags.DEFINE_float('max_grad_norm', 2.0, 'for gradient clipping')
tf.app.flags.DEFINE_float('keep_dropout_prob', 0.5, 'Keep dropout prob')
tf.app.flags.DEFINE_string('filter_sizes', '2,3,5', 'filter_sizes of kernels')
tf.app.flags.DEFINE_integer("early_stop", 10, 'early stop')
tf.app.flags.DEFINE_integer("num_filters", 75, 'filter number')

# TextRNN params
tf.app.flags.DEFINE_integer('rnn_hidden_size', 100, 'dimension of lstm hidden state')

FLAGS = tf.app.flags.FLAGS


def get_config():
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = FLAGS.gpu_fraction
    return config


def run_dev_simple(hps, sess, corpus_data, model, word_vocab, tag_vocab):
    all_sententces = []
    all_predictions = []
    all_labels = []
    all_ids = []

    model.eval = True
    for index, batch in enumerate(minibatches(corpus_data, hps.batch_size)):
        id_batch, word_ids, tags = batch
        if hps.model == 'transformer' or hps.model == 'text_cnn' or hps.model == 'text_rnn_and_cnn':
            word_ids, seq_lens = _pad_sequences(word_ids, 0, hps.max_seq_len)
        else:
            word_ids, seq_lens = pad_sequences(word_ids, 0)

        results = model.run_eval_step(sess, word_ids, tags, seq_lens, [1.0, 1.0, 1.0])
        all_predictions += results['predictions'].tolist()
        all_labels += tags
        all_ids += id_batch
        for word_id in word_ids:
            all_sententces.append(word_id)

    macro_f1 = calc_metrics(hps, all_ids, all_predictions, all_labels, word_vocab, tag_vocab)

    return macro_f1


def run_all(model, train_data, dev_data, test_data, word_vocab, tag_vocab, hps):
    model.build_graph()
    saver = tf.train.Saver(max_to_keep=3)
    sess = tf.Session(config=get_config())
    sess.run(tf.global_variables_initializer())
    summary_writer = tf.summary.FileWriter(hps.log_root)
    bestmodel_save_path = os.path.join(hps.log_root, "bestmodel")
    best_f1, best_acc = None, None
    test_f1 = 1e-8
    not_increse = 0
    result_fop = open(os.path.join(hps.log_root, 'result'), 'w', encoding='utf8')

    config_fop = open(os.path.join(hps.log_root, 'config.json'), 'w', encoding='utf8')
    json.dump(hps._asdict(), config_fop)

    try:
        for i in range(hps.epoch_num):
            for index, batch in enumerate(minibatches(train_data, hps.batch_size)):
                id_batch, word_ids, tags = batch

                if hps.model == 'transformer' or hps.model == 'text_cnn' or hps.model == 'text_rnn_and_cnn':
                    word_ids, seq_lens = _pad_sequences(word_ids, 0, hps.max_seq_len)
                else:
                    word_ids, seq_lens = pad_sequences(word_ids, 0)

                results = model.run_train_step(sess, word_ids, tags, seq_lens, [0.7, 0.7, 0.5])

                loss, train_step, summaries = results['loss'], results['global_step'], results['summaries']
                # print(data_utils.calc_metrics(results['predictions'], tag_ids, tag_vocab, masks))
                if train_step % hps.print_every == 0:
                    tf.logging.info("Epoch %d, Train Step: %d, Loss: %f " % (i, train_step, loss))

            dev_f1 = run_dev_simple(hps, sess, dev_data, model, word_vocab, tag_vocab)

            if best_f1 is None or dev_f1 > best_f1:
                test_f1 = run_dev_simple(hps, sess, test_data, model, word_vocab, tag_vocab)
                best_f1 = dev_f1
                tf.logging.info('Saved best model')
                tf.logging.info("Dev --- \nF1 {:.6f}".format(dev_f1))
                tf.logging.info("Test --- \nF1 {:.6f}".format(test_f1))

                saver.save(sess, bestmodel_save_path, global_step=train_step)
                not_increse = 0
            else:
                not_increse += 1

            if not_increse > hps.early_stop:
                tf.logging.info("Dev ---\nF1 {:.6f}".format(best_f1))
                tf.logging.info("Test ----\nF1 {:.6f}".format(test_f1))
                result_fop.write("Dev ---\nF1 {:.6f}\n".format(best_f1))
                result_fop.write("Test ---\nF1 {:.6f}\n".format(test_f1))

                tf.logging.info('Early Stop with not increasing {:d} batches'.format(hps.early_stop))
                exit(0)
            summary_writer.flush()

        tf.logging.info("Dev ---\nF1 {:.6f}".format(best_f1))
        tf.logging.info("Test ---\nF1 {:.6f}".format(test_f1))
        result_fop.write("Dev ---\nF1 {:.6f}".format(best_f1))
        result_fop.write("Test ---\nF1 {:.6f}".format(test_f1))
    except KeyboardInterrupt:
        tf.logging.info("Dev ---\nF1 {:.6f}".format(best_f1))
        tf.logging.info("Test ---\nF1 {:.6f}".format(test_f1))
        result_fop.write("Dev ---\nF1 {:.6f}".format(best_f1))
        result_fop.write("Test ---\nF1 {:.6f}".format(test_f1))
        exit(0)


def main(argv):
    """Main Method"""
    # Set random seed
    tf.set_random_seed(FLAGS.lucky_num)
    random.seed(FLAGS.lucky_num)

    # Set log config
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.logging.info("Starting {} Model in {} mode".format(FLAGS.model, FLAGS.mode))
    FLAGS.log_root = os.path.join(FLAGS.log_root, FLAGS.exp_name)

    tf.logging.info("Loading vocabulary...")
    word_vocab = Vocab(FLAGS.word_vocab_file, 10000000, True, True, False, False)
    tag_vocab = Vocab(FLAGS.tag_vocab_file, 10000000, False, False, False, False)
    tf.logging.info("Vocabulary loaded!")

    tf.logging.info('Loading data...')
    train_data = SemEval10task8Dataset(FLAGS.train_file, word_vocab, tag_vocab, shuffle=FLAGS.shuffle)
    dev_data = SemEval10task8Dataset(FLAGS.dev_file, word_vocab, tag_vocab, shuffle=False)
    test_data = SemEval10task8Dataset(FLAGS.test_file, word_vocab, tag_vocab, shuffle=False)
    tf.logging.info('Data Loaded!')

    hps_dict = {}
    for key, value in FLAGS.__dict__['__flags'].items():
        tf.logging.info("Parameters in FLAGS: %s - %s" % (key, value))
        hps_dict[key] = value

    hps_dict['filter_sizes'] = [int(filter_size) for filter_size in hps_dict['filter_sizes'].split(',')]
    hps_dict['vocab_size'] = word_vocab.size()
    hps_dict['num_class'] = tag_vocab.size()

    hps = namedtuple("HParams", hps_dict.keys())(**hps_dict)
    tf.logging.info("Word Vocab Size: %d." % word_vocab.size())
    tf.logging.info("Class Number Size: %d." % tag_vocab.size())

    if hps.model == 'attn_lstm':
        model = AttnBiLSTM(hps)
    else:
        raise NotImplementedError('not implemented model: {}'.format(hps.model))

    if FLAGS.mode == 'all':
        run_all(model, train_data, dev_data, test_data, word_vocab, tag_vocab, hps)
    else:
        tf.logging.error('No Supply mode, Please set mode in [train, eval, test, w2v_matrix]')
        exit(-1)


if __name__ == '__main__':
    tf.app.run()
