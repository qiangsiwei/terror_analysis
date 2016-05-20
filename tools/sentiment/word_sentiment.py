# -*- coding: utf-8 -*-

import struct
import fileinput
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

# #### #### 备选词典 #### ####
# exist = {}
# for line in fileinput.input("data/pos_eva.sort.txt"):
# 	exist[line.strip()] = True
# fileinput.close()
# for line in fileinput.input("data/pos_emo.sort.txt"):
# 	exist[line.strip()] = True
# fileinput.close()
# for line in fileinput.input("data/neg_eva.sort.txt"):
# 	exist[line.strip()] = True
# fileinput.close()
# for line in fileinput.input("data/neg_emo.sort.txt"):
# 	exist[line.strip()] = True
# fileinput.close()
# minsim, newword = 0.4, {}
# vocab, feature = load_data(open("../../data/word2vec/vectors.weibo.bin", 'rb'))
# print len(exist.keys())
# count = 0
# for target,value in exist.iteritems():
#     count += 1
#     print count
#     rank = calc_distance(target, vocab, feature)
#     if rank is None:
#         continue
#     indexed_rank = []
#     for i, r in enumerate(rank):
#         indexed_rank.append((r, i))
#     for r in sorted(indexed_rank, key=lambda x: x[0], reverse=True)[1:N]:
#         distance, i = r
#         if distance >= minsim and not exist.has_key(vocab[i]) and not newword.has_key(vocab[i]):
#             newword[vocab[i]] = [target, distance]
# with open("data/addition.txt","w") as f:
#     for k,v in newword.iteritems():
#         try:
#             k.decode("utf-8")
#             file.write(k+"\t"+v[0]+"\t"+str(v[1])+"\n")
#         except:
#             continue

#### #### 极性判断 #### ####
vectormap = {}
for line in fileinput.input("../../data/word2vec/vectors.weibo.txt"):
    try:vectormap[line.strip().split("\t")[0]] = [float(i) for i in line.strip().split("\t")[1].split(" ")]
    except:continue
fileinput.close()
from sklearn import pipeline
from sklearn import linear_model
from sklearn.svm import SVC
from sklearn.cross_validation import KFold
from sklearn.cross_validation import LeaveOneOut
X, y = [], []
for line in fileinput.input("data/pos_eva.sort.txt"):
    if vectormap.has_key(line.strip()):
        X.append(vectormap[line.strip()])
        y.append(0)
fileinput.close()
for line in fileinput.input("data/pos_emo.sort.txt"):
    if vectormap.has_key(line.strip()):
        X.append(vectormap[line.strip()])
        y.append(0)
fileinput.close()
for line in fileinput.input("data/neg_eva.sort.txt"):
    if vectormap.has_key(line.strip()):
        X.append(vectormap[line.strip()])
        y.append(1)
fileinput.close()
for line in fileinput.input("data/neg_emo.sort.txt"):
    if vectormap.has_key(line.strip()):
        X.append(vectormap[line.strip()])
        y.append(1)
X, y = np.array(X), np.array(y)
total, right = 0, 0
# clf = SVC(kernel='linear', class_weight='auto')
# clf = SVC(kernel='poly', degree=3, class_weight='auto')
clf = SVC(kernel='rbf', gamma=1.0, class_weight='auto', probability=True)
clf.fit(X, y)
# for i in xrange(len(X)):
#     total += 1
#     if clf.predict(X[i]) == y[i]:
#         right += 1
# print float(right)/total
with open("data/addition_prob.txt","w") as f:
    for line in fileinput.input("data/addition.txt"):
        word = line.strip().split("\t")[0]
        if word != "":
            rest = clf.predict_proba(vectormap[word]).flatten()
            if any([i>=0.9 for i in rest]):
                file.write(word+"\t"+str(rest[0])+"\t"+str(rest[1])+"\n")
    fileinput.close()
