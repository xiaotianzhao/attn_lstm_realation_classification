#!/usr/bin/python
# -*- coding:utf8 -*-

"""
    @author:xiaotian zhao
    @time:10/28/18
"""

import re
import os
import codecs
import random
import tempfile
import numpy as np
import subprocess

from NRE.utils.vocab import Vocab

PAD_TOKEN = '<PAD>' # This has a vocab id, which is used to pad the encoder input, decoder input and target sequence
UNKNOWN_TOKEN = '<UNK>' # This has a vocab id, which is used to represent out-of-vocabulary words
START_DECODING = '<SOS>' # This has a vocab id, which is used at the start of every decoder input sequence
STOP_DECODING = '<EOS>' # This has a vocab id, which is used at the end of untruncated target sequences


class SemEval10task8Dataset(object):
    """
        Class that iterates over FSAUOR2018 Dataset
        __iter__ method yields a tuple(words, tags)
            words: list of raw words
            tags: list of raw tags
        If processing_word and processing_tag are not None,
        optional preprocessing is applied

        Example:
            data = MsraDataset(file)
            for sentence,tags in data:
                pass
        """

    def __init__(self, filename, word_vocab, tag_vocab, shuffle=True, random_seed=666):
        """
            Args:
                filename: path to the file
                processing_words: (optional) function that takes a word as input
                processing_tags: (optional) function that takes a tag as input
                max_iter: (optional) max number of sentences to yield
            """
        self.filename = filename
        self.word_vocab = word_vocab
        self.tag_vocab = tag_vocab
        self.shuffle = shuffle
        self.length = None
        self.one_hots = np.zeros(shape=[tag_vocab.size(), tag_vocab.size()], dtype=np.float32)
        np.fill_diagonal(self.one_hots, 1)

        random.seed(random_seed)

    def __iter__(self):
        with open(self.filename, encoding="utf-8") as f:
            lines = f.readlines()

        if self.shuffle:
            random.shuffle(lines)

        for line in lines:
            try:
                id, words, target = line.strip().split('\t')
                words = [self.word_vocab.word2id(word) for word in words.split()]
                target = self.tag_vocab.word2id(target)
                yield id, words, target
            except Exception as e:
                print(e)
                print("Error format: %s.", line.strip().split('\t'))

    def __len__(self):
        """
        Iterates once over the corpus to set and store length
        """
        if self.length is None:
            self.length = 0
            for _ in self:
                self.length += 1

        return self.length


def minibatches(data, minibatch_size):
    """
    Args:
        data: generator of (sentence, tags) tuples
        minibatch_size: (int)
    Returns:
        list of tuples
    """
    id_batch, x_batch, y_batch = [], [], []
    for (sen_id, x, y) in data:
        if len(x_batch) == minibatch_size:
            yield id_batch, x_batch, y_batch
            id_batch, x_batch, y_batch = [], [], []
        x_batch += [x]
        y_batch += [y]
        id_batch += [sen_id]

    if len(x_batch) != 0:
        yield id_batch, x_batch, y_batch


def _pad_sequences(sequences, pad_tok, max_length):
    """
        Args:
            sequences: a generator of list or tuple
            pad_tok: the char to pad with
        Returns:
            a list of list where each sublist has same length
        """
    sequence_padded, sequence_length = [], []

    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_length] + [pad_tok] * max(max_length - len(seq), 0)
        sequence_padded += [seq_]
        sequence_length += [min(len(seq), max_length)]

    return sequence_padded, sequence_length


def pad_sequences(sequences, pad_tok):
    """
        Args:
            sequences: a generator of list or tuple
            pad_tok: the char to pad with
        Returns:
            a list of list where each sublist has same length
        """
    max_length = max(map(lambda x: len(x), sequences))
    sequence_padded, sequence_length = _pad_sequences(sequences,
                                                      pad_tok, max_length)
    return sequence_padded, sequence_length


def calc_metrics(hps, ids, predictions, labels, word_vocab, tag_vocab):
    if hps.output_pred and hps.output_gold:
        output_pred_path = hps.output_pred
        output_gold_path = hps.output_gold
        fpo_pred = codecs.open(output_pred_path, 'wb', encoding='utf8')
        fpo_gold = codecs.open(output_gold_path, 'wb', encoding='utf8')
    else:
        descriptor_pred, output_pred_path = tempfile.mkdtemp(suffix='.pred.tmp')
        descriptor_gold, output_gold_path = tempfile.mkdtemp(suffix='.gold.tmp')
        fpo_pred = codecs.getwriter('utf8')(os.fdopen(descriptor_pred, 'wb'))
        fpo_gold = codecs.getwriter('utf8')(os.fdopen(descriptor_gold, 'wb'))

    for id, prediction, label in zip(ids, predictions, labels):
        print('{}\t{}'.format(
            id,
            tag_vocab.id2word(prediction),
        ), file=fpo_pred)

        print('{}\t{}'.format(
            id,
            tag_vocab.id2word(label)
        ), file=fpo_gold)

    fpo_pred.close()
    fpo_gold.close()

    script_args = ['perl', hps.script, output_pred_path, output_gold_path]
    p = subprocess.Popen(script_args, stdout=subprocess.PIPE)
    p.wait()
    std_results = p.stdout.readlines()

    std_result = str(std_results[-1])

    pattern = re.compile('\d+\.\d+')
    macro_f1 = float(pattern.findall(std_result)[0])

    os.remove(output_pred_path)
    os.remove(output_gold_path)
    return macro_f1


if __name__ == '__main__':
    word_vocab = Vocab('../data/word_vocab', 1000000, True, True, False, False)
    target_vocab = Vocab('../data/tag_vocab', 1000, False, False, False, False)
    dataset = SemEval10task8Dataset('../data/valid.txt', word_vocab, target_vocab, shuffle=False)

    count = 0
    for id_batch, words, target in minibatches(dataset, 10):
        # for word_id_list in _pad_sequences(words, word_vocab.word2id(PAD_TOKEN), 1400)[0]:
        #     print(len(word_id_list))
        #
        # print(count, '=========================')
        # count += 1
        print(id_batch)

    # script_args = ['perl', '../data/semeval2010_task8_scorer-v1.2.pl', '../data/proposed_answer1.txt', '../data/answer_key1.txt']
    # p = subprocess.Popen(script_args, stdout=subprocess.PIPE)
    # p.wait()
    # std_results = p.stdout.readlines()
    # std_result = str(std_results[-1])
    # pattern = re.compile(r'\d+\.\d+')
    # print(pattern.findall(std_result)[0])


