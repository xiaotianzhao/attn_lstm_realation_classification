#!/usr/bin/python3
#-*- coding:utf8 -*-


UNK_TOKEN = '<UNK>'
PAD_TOKEN = '<PAD>'

import pickle
import random
import argparse
import numpy as np

if __name__ == '__main__':
    np.random.seed(666)
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_path', type=str, default='', help='embedding file path')
    parser.add_argument('--vocab_path', type=str, default='', help='vocabulary file path')
    parser.add_argument('--vocab_size', type=int, default=1000000000, help='vocabulary size')
    args = parser.parse_args()

    word_embeddings = {}
    w2v_matrix = []
    word_embedding_size = -1
    # load all embeddings
    with open(args.embedding_path, 'r', encoding='utf8') as w_f:
        for index, line in enumerate(w_f):
            err_cnt = 0
            if index >= 0:
                items = line.strip().split()
                word = items[0]
                try:
                    embedding = [float(item) for item in items[1:]]
                    word_embedding_size = len(embedding)
                    word_embeddings[word] = embedding
                except Exception as e:
                    err_cnt += 1

    w2v_matrix.append([0.0 for i in range(word_embedding_size)])

    # unk_vec = [random.random() / 10 for i in range(word_embedding_size)]
    unk_vec = np.random.normal(0, 0.05, word_embedding_size).tolist()
    w2v_matrix.append(unk_vec)
    all_count = 0
    find_count = 0
    # lookup
    with open(args.vocab_path, 'r', encoding='utf8') as w_f:
        for index, line in enumerate(w_f):
            if all_count < args.vocab_size:
                all_count += 1
                word, count = line.strip().split('\t')
                if word in word_embeddings:
                    find_count += 1
                    w2v_matrix.append(word_embeddings[word])
                else:
                    unk_vec = np.random.normal(0, 0.1, word_embedding_size).tolist()
                    # unk_vec = [random.random() / 10 for i in range(word_embedding_size)]
                    w2v_matrix.append(unk_vec)

    pickle.dump(w2v_matrix, open('w2v_matrix', 'wb'), protocol=2)
    print('find rate: %f' % ((1.0 * find_count) / all_count))
    print('Error Count: {}'.format(err_cnt))
    print(word_embedding_size)
