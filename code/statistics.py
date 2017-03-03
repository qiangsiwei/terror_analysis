# -*- coding: utf-8 -*-

import time
import gzip
import fileinput

# 时间跨度筛选
l1, l2 = [0], [0]
for line in gzip.open("data/original/t_lable_group_comp_1.sort.seg.txt.gz"):
	day = (int(time.mktime(time.strptime(line.strip().split("\t")[1][1:-1],'%Y-%m-%d %H:%M:%S')))-int(time.mktime(time.strptime("2013_01-01 00:00:00",'%Y-%m-%d %H:%M:%S'))))/(24*3600)
	if day >= 0:
		l1.append(day)
for line in gzip.open("data/original/t_lable_group_comp_4.sort.seg.txt.gz"):
	day = (int(time.mktime(time.strptime(line.strip().split("\t")[1][1:-1],'%Y-%m-%d %H:%M:%S')))-int(time.mktime(time.strptime("2013_01-01 00:00:00",'%Y-%m-%d %H:%M:%S'))))/(24*3600)
	if day >= 0:
		l2.append(day)
import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(8, 4))
ax1.hist(l1, 500, normed=1, histtype='stepfilled', facecolor='g', rwidth=0.8)
ax2.hist(l2, 500, normed=1, histtype='stepfilled', facecolor='b', rwidth=0.8)
plt.tight_layout()
plt.show()

# 日志条数统计
map1, map2 = {}, {}
for line in gzip.open("data/original/t_lable_group_comp_1.sort.seg.txt.gz"):
	day = (int(time.mktime(time.strptime(line.strip().split("\t")[1][1:-1],'%Y-%m-%d %H:%M:%S')))-int(time.mktime(time.strptime("2013_01-01 00:00:00",'%Y-%m-%d %H:%M:%S'))))/(24*3600)
	if day >= 0:
		key = line.strip().split("\t")[1][1:8]
		map1[key] = 1 if not map1.has_key(key) else map1[key]+1
for line in gzip.open("data/original/t_lable_group_comp_4.sort.seg.txt.gz"):
	day = (int(time.mktime(time.strptime(line.strip().split("\t")[1][1:-1],'%Y-%m-%d %H:%M:%S')))-int(time.mktime(time.strptime("2013_01-01 00:00:00",'%Y-%m-%d %H:%M:%S'))))/(24*3600)
	if day >= 0:
		key = line.strip().split("\t")[1][1:8]
		map2[key] = 1 if not map2.has_key(key) else map2[key]+1
print map1, map2

# 事件发生次数统计
stat = {}
for line in fileinput.input("data/events/classified_news.txt"):
	tm, eid, tag = line.split("\t")[3][0:7], line.split("\t")[0], line.split("\t")[1]
	if not stat.has_key(tm):
		stat[tm] = {}
	if not stat[tm].has_key(tag):
		stat[tm][tag] = {}
	if not stat[tm][tag].has_key(eid):
		stat[tm][tag][eid] = True
fileinput.close()
for k,v in stat.iteritems():
	print k, len(v["1"].keys()), len(v["2"].keys()), len(v["3"].keys())

# 2013_01: 20 32 47 16214
# 2013_02: 9 22 25 8608
# 2013_03: 18 30 32 9877
# 2013_04: 30 59 37 16840
# 2013_05: 30 62 52 13640
# 2013_06: 88 58 39 36617
# 2013_07: 42 148 34 28855
# 2013_08: 43 74 34 18434 3495
# 2013_09: 37 45 32 15473 2235
# 2013_10: 41 67 27 12116 9300
# 2013_11: 26 74 43 19190 5476
# 2013_12: 26 63 34 15939 7604
# 2014_01: 23 51 25 13650 3509
# 2014_02: 36 27 24 13885 7982
# 2014_03: 61 364 51 96967 96407
# 2014_04: 32 98 33 22495 10827

# 筛选出待标注数据
with open("data/testset/classified_news.txt","w") as f:
	for line in fileinput.input("data/events/version/classified_news.txt"):
		if line.split("\t")[3][0:7] in ["2013_10"]:
			f.write("\t".join(line.strip().split("\t")[0:5])+"\n")
	fileinput.close()

with open("data/testset/classified_weibo.txt","w") as f:
	for line in fileinput.input("data/events/version/classified_weibo.txt"):
		if line.split("\t")[3][0:7] in ["2013_10"]:
			f.write("\t".join(line.strip().split("\t")[0:5])+"\n")
	fileinput.close()

with open("data/testset/tag_neg_t_lable_group_comp_1.seg.txt","w") as f:
	for line in fileinput.input("data/classify/tag_neg_t_lable_group_comp_1.seg.txt"):
		if line.split("\t")[0][0:7] in ["2013_10"]:
			f.write("\t".join(line.split("\t")[1:3])+"\t"+line.split("\t")[0]+"\t"+line.split("\t")[-1])
	fileinput.close()

with open("data/testset/tag_neg_t_lable_group_comp_4.seg.txt","w") as f:
	for line in fileinput.input("data/classify/tag_neg_t_lable_group_comp_4.seg.txt"):
		if line.split("\t")[0][0:7] in ["2013_10"]:
			f.write("\t".join(line.split("\t")[0:3])+"\t"+line.split("\t")[0]+"\t"+line.split("\t")[-1])
	fileinput.close()

# 筛选出待标注数据(news)
c, cmap = 0, {}
for line in gzip.open("data/original/t_lable_group_comp_1.sort.seg.txt.gz"):
	c += 1
	print c
	cid, title, content = line.split("\t")[0][1:-1], line.split("\t")[4][1:-1], line.split("\t")[7][1:-1]
	cmap[cid] = title+"\n"
with open("data/testset/news.txt","w") as f:
	emap = {}
	for line in fileinput.input("data/events/version/classified_news.txt"):
		tm, eid, tag, cid = line.split("\t")[3][0:7], line.split("\t")[0], line.split("\t")[1], line.split("\t")[3]
		if tm in ["2013_08","2013_09","2013_10"]:	
			emap[eid] = 1 if not emap.has_key(eid) else emap[eid]+1
			if emap[eid] <= 5:
				f.write(tag+"\t"+eid+"\t"+cid+"\t"+cmap[cid])
	fileinput.close()

# 筛选出待标注数据(weibo)
c, cmap = 0, {}
for line in gzip.open("data/original/t_lable_group_comp_4.sort.seg.txt.gz"):
	c += 1
	print c
	cid, content = line.split("\t")[0][1:-1], line.split("\t")[5][1:-1]
	cmap[cid] = content+"\n"
with open("data/testset/weibo.txt","w") as f:
	emap = {}
	for line in fileinput.input("data/events/version/classified_weibo.txt"):
		tm, eid, tag, cid = line.split("\t")[3][0:7], line.split("\t")[0], line.split("\t")[1], line.split("\t")[3]
		if tm in ["2013_08","2013_09","2013_10"]:
			emap[eid] = 1 if not emap.has_key(eid) else emap[eid]+1
			if emap[eid] <= 5:
				try:
					cmap[cid].decode("utf-8")
					f.write(tag+"\t"+eid+"\t"+cid+"\t"+cmap[cid])
				except:
					continue
	fileinput.close()

# weibo筛选出标签
with open("data/events/classified_weibo_hashtag.txt","w") as f:
	import re
	pattern = re.compile(r'#(.*?)#')
	for line in fileinput.input("data/events/classified_weibo.txt"):
		eid, cid, content = line.split("\t")[0], line.split("\t")[3], line.split("\t")[4]
		match = pattern.match(content)
		if match:
			f.write(eid+"\t"+cid+"\t"+match.group()+"\n")
	fileinput.close()

# 筛选出待标注数据(CRF)
c, cmap = 0, {}
for line in gzip.open("data/original/t_lable_group_comp_1.sort.seg.txt.gz"):
	c += 1
	print c
	cid, title, content = line.split("\t")[0][1:-1], line.split("\t")[4][1:-1], line.split("\t")[7][1:-1]
	cmap[cid] = title+"\n"
with open("data/testset/crf.txt","w") as f:
	emap = {}
	for line in fileinput.input("data/events/version/classified_news.txt"):
		tm, eid, cid = line.split("\t")[3][0:7], line.split("\t")[0], line.split("\t")[3]
		if tm in ["2013_10","2013_11","2013_12"]:
			emap[eid] = 1 if not emap.has_key(eid) else emap[eid]+1
			if emap[eid] <= 3:
				file.write(cmap[cid])
	fileinput.close()

emap = {1:{},2:{},3:{}}
for line in fileinput.input("data/events/version/classified_news.txt"):
	part = line.strip().split("\t")
	e, c, d = int(part[0]), int(part[1]), int(part[2])
	if not emap[c].has_key(e):
		emap[c][e] = {"news":[d],"weibo":[]}
	else:
		emap[c][e]["news"].append(d)
fileinput.close()

for line in fileinput.input("data/events/version/classified_weibo.txt"):
	part = line.strip().split("\t")
	e, c, d = int(part[0]), int(part[1]), int(part[2])
	emap[c][e]["weibo"].append(d)
fileinput.close()

import matplotlib.pyplot as plt
for k,v in emap.iteritems():
	l1, l2, l3 = [0,1125], [0,1125], [0,1125]
	for e,r in v.iteritems():
		l1.append(min(r["news"]))
		l2.extend(r["news"])
		l3.extend(r["weibo"])
	fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(8, 4))
	ax0.hist(l1, 500, normed=1, histtype='bar', facecolor='r', alpha=0.75)
	ax0.set_title('type '+str(k)+' event density')
	ax1.hist(l2, 500, normed=1, histtype='stepfilled', facecolor='g', rwidth=0.8)
	ax1.set_title('type '+str(k)+' webnews density')
	ax2.hist(l3, 500, normed=1, histtype='stepfilled', facecolor='b', rwidth=0.8)
	ax2.set_title('type '+str(k)+' sinaweibo density')
	plt.tight_layout()
	plt.show()
