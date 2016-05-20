# -*- coding: utf-8 -*-

import fileinput
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import KFold
from gensim import corpora, models, similarities

documents = []
for line in fileinput.input("../../data/testset/news_seed.txt"):
	documents.append(" ".join([item[0] for item in filter(lambda x:x[1] in ["a","an","b","n","v","vn","vi"], [item.split("/") for item in line.strip().split("\t")[1].split(" ")])]))
fileinput.close()
texts = [[word for word in document.lower().split()] for document in documents]
dictionary = corpora.Dictionary(texts)
for k,v in dictionary.token2id.iteritems():
	dictionary.id2token[v] = k
corpus = [dictionary.doc2bow(text) for text in texts]
X, y = [], []
for i in xrange(len(corpus)):
	cmap = {}
	for k in corpus[i]:
		cmap[k[0]] = k[1]
	X.append([0 if not cmap.has_key(j) else cmap[j] for j in xrange(len(dictionary.keys()))])
	y.append(1 if 0<=i<20 else 2 if 20<=i<40 else 3 if 40<=i<60 else 0)
X, y = np.array(X), np.array(y)
total, right = 0, 0
for t in xrange(10):
	kf = KFold(len(X), n_folds=10)
	for train, test in kf:
		clf = MultinomialNB()
		clf.fit(X[train], y[train])
		for item in test:
			total += 1
			if clf.predict(X[item]) == y[item]:
				right += 1
print float(right)/total
# 0.68

documents = []
for line in fileinput.input("../../data/testset/weibo_seed.txt"):
	documents.append(" ".join([item[0] for item in filter(lambda x:x[1] in ["a","an","b","n","v","vn","vi"], [item.split("/") for item in line.strip().split(" ")])]))
fileinput.close()
texts = [[word for word in document.lower().split()] for document in documents]
dictionary = corpora.Dictionary(texts)
for k,v in dictionary.token2id.iteritems():
	dictionary.id2token[v] = k
corpus = [dictionary.doc2bow(text) for text in texts]
X, y = [], []
for i in xrange(len(corpus)):
	cmap = {}
	for k in corpus[i]:
		cmap[k[0]] = k[1]
	X.append([0 if not cmap.has_key(j) else cmap[j] for j in xrange(len(dictionary.keys()))])
	y.append(1 if 0<=i<20 else 2 if 20<=i<40 else 3 if 40<=i<60 else 0)
X, y = np.array(X), np.array(y)
total, right = 0, 0
for t in xrange(10):
	kf = KFold(len(X), n_folds=10)
	for train, test in kf:
		clf = MultinomialNB()
		clf.fit(X[train], y[train])
		for item in test:
			total += 1
			if clf.predict(X[item]) == y[item]:
				right += 1
print float(right)/total
# 0.71
