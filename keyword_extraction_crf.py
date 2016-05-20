# -*- coding: utf-8 -*-

import gzip
import fileinput
import pycrfsuite

def word2features(sent, i):
	word = sent[i][0]
	postag = sent[i][1]
	features = [
		'bias',
		'word=' + word,
		'word.isdigit=%s' % word.isdigit(),
		'+2:postag=' + postag,
	]
	if i > 0:
		word1 = sent[i-1][0]
		postag1 = sent[i-1][1]
		features.extend([
			'+1:word=' + word1,
			'+1:postag=' + postag1,
		])
	else:
		features.append('BOS')
	if i < len(sent)-1:
		word1 = sent[i+1][0]
		postag1 = sent[i+1][1]
		features.extend([
			'+1:word=' + word1,
			'+1:postag=' + postag1,
		])
	else:
		features.append('EOS')
	if i < len(sent)-2:
		word1 = sent[i+2][0]
		postag1 = sent[i+2][1]
		features.extend([
			'+1:word=' + word1,
			'+1:postag=' + postag1,
		])
	if i < len(sent)-3:
		word1 = sent[i+3][0]
		postag1 = sent[i+3][1]
		features.extend([
			'+1:word=' + word1,
			'+1:postag=' + postag1,
		])
	if i < len(sent)-4:
		word1 = sent[i+4][0]
		postag1 = sent[i+4][1]
		features.extend([
			'+1:word=' + word1,
			'+1:postag=' + postag1,
		]) 
	return features

def sent2features(sent):
	return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
	return [label for token, postag, label in sent]

def sent2tokens(sent):
	return [item[0] for item in sent]

# 验证准确率与召回率
sents = []
for line in fileinput.input("data/testset/crf_train.txt"):
	items = [item.split("/") for item in line.strip().split(" ")]
	for item in items:
		if len(item) == 2:item.append("NL")
	sents.append(items)

X_train = [sent2features(s) for s in sents[1::2]]
y_train = [sent2labels(s) for s in sents[1::2]]
trainer = pycrfsuite.Trainer(verbose=False)
for xseq, yseq in zip(X_train, y_train):
	trainer.append(xseq, yseq)
trainer.set_params({
	'c1': 1.0,
	'c2': 1e-3,
	# 'max_iterations': 50,
	'feature.possible_transitions': True
})
trainer.params()
trainer.train('data/statextract.crfsuite')
tagger = pycrfsuite.Tagger()
tagger.open('data/statextract.crfsuite')

t1, t2, t3 = 0, 0, 0
for i in xrange(len(sents[0::2])):
	sent_test, s1, s2 = sents[0::2][i], [], []
	taged, labeled = tagger.tag(sent2features(sent_test)), sent2labels(sent_test)
	# print ' '.join(sent2tokens(sent_test))
	# print taged
	# print labeled
	# print tagger.probability(taged)
	for j in xrange(len(taged)):
		if taged[j] == "TG":s1.append(j)
		if labeled[j] == "TG":s2.append(j)
	t1, t2, t3 = t1+len(s1), t2+len(s2), t3+len(list(set(s1).intersection(set(s2))))
# print t1, t2, t3, float(t3)/t1, float(t3)/t2
print round(float(t3)/t1,4), round(float(t3)/t2,4)

# 协同训练
import copy
testsents, N = sents, len(sents)/2
# testsents, N = copy.copy(sents), len(sents)/2
for i in xrange(5):
	probs = []
	for line in fileinput.input("data/testset/crf.txt"):
		title = [item.split("/") for item in line.strip().split(" ")]
		title_taged = tagger.tag(sent2features(title))
		if "TG" in title_taged:
			title_train_list = [[title[i][0],title[i][1],title_taged[i]] for i in xrange(len(title))]
			title_taged_prob = tagger.probability(title_taged)
			probs.append({"title_train_list":title_train_list,"title_taged_prob":title_taged_prob})
	fileinput.close()
	probs = [prob["title_train_list"] for prob in sorted(probs, key=lambda x: x["title_taged_prob"], reverse=True)[0:N]]
	testsents.extend(probs)

	X_train = [sent2features(s) for s in testsents[1::2]]
	y_train = [sent2labels(s) for s in testsents[1::2]]
	trainer = pycrfsuite.Trainer(verbose=False)
	for xseq, yseq in zip(X_train, y_train):
		trainer.append(xseq, yseq)
	trainer.set_params({
		'c1': 1.0,
		'c2': 1e-3,
		# 'max_iterations': 50,
		'feature.possible_transitions': True
	})
	trainer.params()
	trainer.train('data/statextract.crfsuite')
	tagger = pycrfsuite.Tagger()
	tagger.open('data/statextract.crfsuite')

	t1, t2, t3 = 0, 0, 0
	for i in xrange(len(sents[0::2])):
		sent_test, s1, s2 = sents[0::2][i], [], []
		taged, labeled = tagger.tag(sent2features(sent_test)), sent2labels(sent_test)
		for j in xrange(len(taged)):
			if taged[j] == "TG":s1.append(j)
			if labeled[j] == "TG":s2.append(j)
		t1, t2, t3 = t1+len(s1), t2+len(s2), t3+len(list(set(s1).intersection(set(s2))))
	# print t1, t2, t3, float(t3)/t1, float(t3)/t2
	print round(float(t3)/t1,4), round(float(t3)/t2,4)
# 0.8621 0.9434
# 0.9024 0.9610
# 0.9151 0.9700
# 0.9313 0.9760
# 0.9416 0.9797
# 0.9497 0.9827

# # 验证全集中的抽取率
# c = 0
# cmap = {}
# for line in gzip.open("data/original/t_lable_group_comp_1.sort.seg.txt.gz"):
# 	cid, title, content = line.split("\t")[0][1:-1], line.split("\t")[4], line.split("\t")[7]
# 	cmap[cid] = [title, content]
# 	c += 1
# 	print c
# c = 0
# emap, total = {}, 0
# for line in fileinput.input("data/events/classified_news.txt"):
# 	c += 1
# 	print c
# 	try:
# 		tm, eid, cid = line.split("\t")[3][0:7], line.split("\t")[0], line.split("\t")[3]
# 		title, content = [item.split("/") for item in cmap[cid][0][1:-1].split(" ")], [item.split("/") for item in cmap[cid][1][1:-1].split(" ")]
# 		title_taged = tagger.tag(sent2features(title)) if cmap[cid][0].replace("\"","").strip()!="" else []
# 		# content_taged = tagger.tag(sent2features(content)) if cmap[cid][1].replace("\"","").strip()!="" else []
# 		if not emap.has_key(tm):emap[tm] = 0
# 		if "TG" in title_taged:
# 			print tagger.probability(title_taged)
# 			emap[tm] = 1
# 			total += 1
# 	except:
# 		continue
# fileinput.close()
# print total, sum([emap[key] for key in emap.keys()])
# # 0.45511765049337305 抽取率
