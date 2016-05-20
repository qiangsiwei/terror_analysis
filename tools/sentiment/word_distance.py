#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import struct
import numpy as np

N = 40  # number of closest words that will be shown
max_w = 50  # max length of vocabulary entries

def load_data(f):
    # file type f
    data = f.read(max_w)
    sep = data.find('\n')
    word_n_size = data[:sep]
    words, size = word_n_size.split()
    words, size = int(words), int(size)
    vocab = []
    feature = []
    data = data[sep+1:]
    for b in range(words):
        c_data = f.read(max_w + 1)
        data = data + c_data
        separator = data.find(' ')
        w = data[:separator]
        vocab.append(w)
        data = data[separator+1:]
        if len(data) < 4*size:  # assuming 4 byte float
            data += f.read(4*size)
        vec = np.array(struct.unpack("{}f".format(size), data[:4*size]))
        length = np.sqrt((vec**2).sum())
        vec /= length
        feature.append(vec)
        data = data[4*size+1:]
    feature = np.array(feature)
    return vocab, feature

def calc_distance(target, vocab, feature):
    try:
        i = vocab.index(target)
        rank = (feature * feature[i]).sum(axis=1)
    except ValueError:
        # target does not exist
        rank = None
    return rank

def load_freq(f):
    freqlist = []
    for line in f:
        word, freq = line.split()
        freqlist.append(freq)
    return freqlist

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python distance.py <FILE>\nwhere FILE contains word projections in the BINARY FORMAT\n")
        sys.exit()
    filename = sys.argv[1]
    try:
        with open(filename, 'rb') as f:
            vocab, feature = load_data(f)
    except IOError:
        print("Input file not found\n")
        sys.exit(-1)
    if len(sys.argv) == 3:
        vocab_filename = sys.argv[2]
        try:
            with open(vocab_filename, 'r') as f:
                freqlist = load_freq(f)
        except IOError:
            print("Input vocab file not found\n")
            sys.exit(-1)
    while True:
        target = raw_input("Enter word: ")
        rank = calc_distance(target, vocab, feature)
        if rank is None:
            print("Out of dictionary word!")
            continue
        indexed_rank = []
        for i, r in enumerate(rank):
            indexed_rank.append((r, i))
        print("word\tdistance\tfrequency")
        for r in sorted(indexed_rank, key=lambda x: x[0], reverse=True)[1:N]:
            distance, i = r
            if len(sys.argv) == 3:
                print("{}\t{:06f}\t{}".format(vocab[i], distance, freqlist[i]))
            else:
                print("{}\t{:06f}".format(vocab[i], distance))
        print("")
