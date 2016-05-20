# -*- coding: utf-8 -*-

import time
import fileinput
from gensim import corpora, models, similarities

event_map = {}
for line in fileinput.input("data/events/classified_news.txt"):
	part = line.strip().split("\t")
	event, cls, day = int(part[0]), int(part[1]), int(part[2])
	nr_t = None if len(part[5].split("|")[0][1:-1]) == 0 else [(item.split(":")[0], int(item.split(":")[-1])) for item in part[5].split("|")[0][1:-1].split(" ")]
	ns_t = None if len(part[5].split("|")[1][1:-1]) == 0 else [(item.split(":")[0], int(item.split(":")[-1])) for item in part[5].split("|")[1][1:-1].split(" ")]
	nr_c = None if len(part[5].split("|")[2][1:-1]) == 0 else [(item.split(":")[0], int(item.split(":")[-1])) for item in part[5].split("|")[2][1:-1].split(" ")]
	ns_c = None if len(part[5].split("|")[3][1:-1]) == 0 else [(item.split(":")[0], int(item.split(":")[-1])) for item in part[5].split("|")[3][1:-1].split(" ")]
	text, weight = [], 5
	if nr_t != None:
		for w in nr_t:
			try:
				text.extend([w[0].decode("utf-8")]*w[1]*weight)
			except:
				continue
	if ns_t != None:
		for w in ns_t:
			try:
				text.extend([w[0].decode("utf-8")]*w[1]*weight)
			except:
				continue
	if nr_c != None:
		for w in nr_c:
			try:
				text.extend([w[0].decode("utf-8")]*min(w[1],3))
			except:
				continue
	if ns_c != None:
		for w in ns_c:
			try:
				text.extend([w[0].decode("utf-8")]*min(w[1],3))
			except:
				continue
	if not event_map.has_key(event):
		event_map[event] = {"cls":cls,"stime":day,"text":text}
	else:
		event_map[event]["text"].extend(text)
fileinput.close()

documents = []
for k, v in event_map.iteritems():
	documents.append(" ".join(v["text"]).encode("utf-8"))

weibo = []
for line in fileinput.input("data/classify/tag_pos_t_lable_group_comp_4.seg.txt"):
	part = line.strip().split("\t")
	weibo.append(part[1]+"\t"+part[2]+"\t"+part[0]+"\t"+part[5])
	text = []
	for item in line.strip().split("\t")[3][1:-1].split(" "):
		if item != "":
			word, freq = ":".join(item.split(":")[:-1]), int(item.split(":")[-1])
			text.extend([word.decode("utf-8")]*freq)
	for item in line.strip().split("\t")[4][1:-1].split(" "):
		if item != "":
			word, freq = ":".join(item.split(":")[:-1]), int(item.split(":")[-1])
			text.extend([word.decode("utf-8")]*freq)
	documents.append(" ".join(text).encode("utf-8"))
fileinput.close()

texts = [[word for word in document.lower().split()] for document in documents]
dictionary = corpora.Dictionary(texts)
print len(dictionary.token2id)
corpus = [dictionary.doc2bow(text) for text in texts]
tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]

file_event = open("data/classify/corpus_tfidf_event.txt","w")
file_weibo = open("data/classify/corpus_tfidf_weibo.txt","w")
c = 0
for doc in corpus_tfidf:
	if c < len(event_map.keys()):
		file_event.write(str(event_map[c]["cls"])+"\t"+str(event_map[c]["stime"])+"\t"+"\t".join([str(item[0])+" "+str(item[1]) for item in doc])+"\n")
	else:
		file_weibo.write(weibo[c-len(event_map.keys())]+"\t"+"\t".join([str(item[0])+" "+str(item[1]) for item in doc])+"\n")
	c += 1
file_event.close()
file_weibo.close()

