#!/usr/bin/python
# -*- coding:utf8 -*-

"""
    @author:xiaotian zhao
    @time:3/20/19
"""

import csv

PAD_TOKEN = '<PAD>' # This has a vocab id, which is used to pad the encoder input, decoder input and target sequence
UNKNOWN_TOKEN = '<UNK>' # This has a vocab id, which is used to represent out-of-vocabulary words
START_DECODING = '<SOS>' # This has a vocab id, which is used at the start of every decoder input sequence
STOP_DECODING = '<EOS>' # This has a vocab id, which is used at the end of untruncated target sequences


class Vocab(object):
    """Vocabulary class for mapping words to ids"""
    def __init__(self, vocab_file, max_size, use_unk=True, use_pad=True, use_start=True, use_end=True):
        self._word2id = {}
        self._id2word = {}
        self._count = 0
        if use_pad:
            self._word2id[PAD_TOKEN] = self._count
            self._count += 1

        if use_unk:
            self._word2id[UNKNOWN_TOKEN] = self._count
            self._count += 1

        if use_start:
            self._word2id[START_DECODING] = self._count
            self._count += 1

        if use_end:
            self._word2id[STOP_DECODING] = self._count
            self._count += 1

        with open(vocab_file, 'r', encoding='utf8') as vocab_f:
            for line in vocab_f:
                items = line.strip().split()
                if len(items) != 2:
                    print('Warning : incorrectly formatted line in vocabulary file: %s\n' % line)
                    continue
                word, _ = items
                if word in self._word2id:
                    raise Exception('Duplicated word in vocabulary')

                self._word2id[word] = self._count
                self._count += 1

                if max_size != 0 and self._count >= max_size:
                    print('max size of vocab was specified as %i; we now have %i words. Stopping reading' % (max_size, self._count))
                    break

        self._id2word = dict(zip(self._word2id.values(), self._word2id.keys()))

    def word2id(self, word):
        if word not in self._word2id:
            return self._word2id[UNKNOWN_TOKEN]
        else:
            return self._word2id[word]

    def id2word(self, word_id):
        if word_id not in self._id2word:
            raise ValueError('Id not found in vocab: %d' % word_id)
        return self._id2word[word_id]

    def size(self):
        return self._count

    def write_metadata(self, fpath):
        """Writes metadata file for Tensorboard word embedding visualizer as described here:
          https://www.tensorflow.org/get_started/embedding_viz

        Args:
          fpath: place to write the metadata file
        """
        print("Writing word embedding metadata file to %s..." % (fpath))
        with open(fpath, "w") as f:
            fieldnames = ['word']
            writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
            for i in range(self.size()):
                writer.writerow({"word": self._id_to_word[i]})