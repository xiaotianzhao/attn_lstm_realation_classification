#!/usr/bin/python
# -*- coding:utf8 -*-

"""
    @author:xiaotian zhao
    @time:11/27/18
"""
import argparse
from collections import Counter

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, default='./train.txt', help='train data file')
    parser.add_argument("--vocab_file", type=str, default='./word_vocab', help='word vocabulary file')
    parser.add_argument("--tag_vocab_file", type=str, default='./tag_vocab', help='tag vocabulary file')

    args = parser.parse_args()

    word_counter = Counter()
    tag_counter = Counter()

    with open(args.data_file, 'r', encoding='utf8') as reader:
        for line in reader:
            _, words, label = line.strip().split('\t')
            words = words.split()
            word_counter.update(words)
            tag_counter.update([label])

    reader.close()

    word_vocab_file = open(args.vocab_file, 'w', encoding='utf8')
    tag_vocab_file = open(args.tag_vocab_file, 'w', encoding='utf8')

    for word, count in word_counter.most_common():
        word_vocab_file.write('%s\t%s\n' % (word, count))

    for word, count in tag_counter.most_common():
        tag_vocab_file.write('%s\t%s\n' % (word, count))
