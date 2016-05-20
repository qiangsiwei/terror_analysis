# -*- coding: utf-8 -*-

import gzip
import fileinput
from gensim import corpora, models, similarities

cmap = {}
for line in gzip.open("data/original/t_lable_group_comp_1.sort.seg.txt.gz"):
	cid, title, content = line.split("\t")[0][1:-1], line.split("\t")[4][1:-1], line.split("\t")[7][1:-1]
	cmap[cid] = title+"\t"+content
with open("data/testset/news_seed.txt","w") as f:
	for line in fileinput.input("data/testset/news_train.txt"):
		if len(line.strip().split("\t")[0]) == 1:
			f.write(cmap[line.split("\t")[2]]+"\n")
		else:
			f.write(cmap[line.split("\t")[0]]+"\n")
	fileinput.close()

cmap = {}
for line in gzip.open("data/original/t_lable_group_comp_4.sort.seg.txt.gz"):
	cid, content = line.split("\t")[0][1:-1], line.split("\t")[5][1:-1]
	cmap[cid] = content
with open("data/testset/weibo_seed.txt","w") as f:
	for line in fileinput.input("data/testset/weibo_train.txt"):
		if len(line.strip().split("\t")[0]) == 1:
			f.write(cmap[line.split("\t")[2]]+"\n")
		else:
			f.write(cmap[line.split("\t")[0]]+"\n")
	fileinput.close()

# 评估详见 tools/tsne/*
