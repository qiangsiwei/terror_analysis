# -*- coding: utf-8 -*-

import math
import gzip
import fileinput
from gensim import corpora, models, similarities

def cos(a, b):
	import math
	try:
		return sum([sa*sb if pa == pb else 0 for (pa,sa) in a for (pb,sb) in b])/math.sqrt(sum([sa**2 for (pa,sa) in a])*sum([sb**2 for (pb,sb) in b]))
	except:
		return 0

documents, news = [], []
for line in fileinput.input("data/classify/tag_pos_t_lable_group_comp_1.seg.txt"):
	part = line.strip().split("\t")
	cid, cls, day = part[0], int(part[1]), int(part[2])
	title_nf = None if len(part[3][1:-1]) == 0 else [(item.split(":")[0], int(item.split(":")[-1])) for item in part[3][1:-1].split(" ")]
	title_ns = None if len(part[4][1:-1]) == 0 else [(item.split(":")[0], int(item.split(":")[-1])) for item in part[4][1:-1].split(" ")]
	content_nf = None if len(part[5][1:-1]) == 0 else [(item.split(":")[0], int(item.split(":")[-1])) for item in part[5][1:-1].split(" ")]
	content_ns = None if len(part[6][1:-1]) == 0 else [(item.split(":")[0], int(item.split(":")[-1])) for item in part[6][1:-1].split(" ")]
	title = part[-1]
	news.append([cls, day, cid+"\t"+title+"\t"+part[3]+"|"+part[4]+"|"+part[5]+"|"+part[6]])
	text, weight, maxct = [], 5, 3
	if title_nf != None:
		for w in title_nf:
			try:
				text.extend([w[0].decode("utf-8")]*w[1]*weight)
			except:
				continue
	if title_ns != None:
		for w in title_ns:
			try:
				text.extend([w[0].decode("utf-8")]*w[1]*weight)
			except:
				continue
	if content_nf != None:
		for w in content_nf:
			try:
				text.extend([w[0].decode("utf-8")]*min(w[1],maxct))
			except:
				continue
	if content_ns != None:
		for w in content_ns:
			try:
				text.extend([w[0].decode("utf-8")]*min(w[1],maxct))
			except:
				continue
	documents.append(" ".join(text).encode("utf-8"))
fileinput.close()
texts = [[word for word in document.lower().split()] for document in documents]
dictionary = corpora.Dictionary(texts)
print len(dictionary.token2id)
corpus = [dictionary.doc2bow(text) for text in texts]
tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]
event = {1:[],2:[],3:[]}
c, mint, mino, assignmap = 0, math.exp(-14), 0.15, {}
for doc in corpus_tfidf:
	print c
	maxsim, assign, feature = 0, -1, doc
	for e in xrange(len(event[news[c][0]])):
		simt = math.exp(event[news[c][0]][e]["stime"] - news[c][1])
		if simt >= mint:
			simo = cos(event[news[c][0]][e]["feature"],feature)
			if simt*simo >= mint*mino and simo*simo > maxsim:
				maxsim, assign = simo*simo, e
	if assign != -1:
		assignmap[c] = assign
		event[news[c][0]][assign]["title"].append(news[c][2])
		fmap = {}
		for (p,s) in event[news[c][0]][assign]["feature"]:
			fmap[p] = s
		for (p,s) in feature:
			fmap[p] = s if not fmap.has_key(p) else fmap[p]+s
		event[news[c][0]][assign]["feature"] = [(p,s) for p,s in fmap.iteritems()]
	else:
		assignmap[c] = len(event[news[c][0]])
		event[news[c][0]].append({"stime":news[c][1],"feature":feature,"title":[news[c][2]]})
	c += 1
with open("data/events/classified_news.txt","w") as f:
	c = 0
	for i in [1,2,3]:
		for e in event[i]:
			for t in e["title"]:
				f.write(str(c)+"\t"+str(i)+"\t"+str(e["stime"])+"\t"+t+"\n")
			c += 1
with open("data/events/classified_feat.txt","w") as f:
	c = 0
	for i in [1,2,3]:
		for e in event[i]:
			f.write(str(c)+"\t"+" ".join([str(p)+":"+str(s) for (p,s) in e["feature"]]))
			c += 1

# change = 0
# for i in xrange(1):
# 	c = 0
# 	for doc in corpus_tfidf:
# 		print c
# 		for e in xrange(len(event[news[c][0]])):
# 			simt = math.exp(-abs(event[news[c][0]][e]["stime"] - news[c][1]))
# 			if simt >= mint:
# 				simo = cos(event[news[c][0]][e]["feature"],feature)
# 				if simt*simo >= mint*mino and simo*simo > maxsim:
# 					maxsim, assign = simo*simo, e
# 		if assign != -1 and assign != assignmap[c]:
# 			change += 1
# 		c += 1
# print "change: ", change
