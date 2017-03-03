# -*- coding: utf-8 -*-

import math
import time
import gzip
import fileinput
import numpy as np
from sklearn import pipeline
from sklearn import linear_model
from sklearn.svm import SVC
from sklearn.cross_validation import KFold
from sklearn.cross_validation import LeaveOneOut

vectormap, vclass1, vclass2, vclass3 = {}, [], [], []
for line in fileinput.input("record.txt"):
	part = line.strip().split("\t")
	vectormap[part[2]] = [float(part[4].split(",")[0]), float(part[4].split(",")[1])]
	if part[1] == "1":
		vclass1.append([float(part[4].split(",")[0]), float(part[4].split(",")[1])])
	if part[1] == "2":
		vclass2.append([float(part[4].split(",")[0]), float(part[4].split(",")[1])])
	if part[1] == "3":
		vclass3.append([float(part[4].split(",")[0]), float(part[4].split(",")[1])])
fileinput.close()
print vclass1, vclass2, vclass3
i, X, y = 0, [], []
for line in fileinput.input("../../data/testset/news_seed.txt"):
	i += 1
	print i
	words_title = [item[0] for item in filter(lambda x:x[1] in ["a","an","b","n","v","vn","vi"], [item.split("/") for item in line.strip().split("\t")[0].split(" ")])]
	s1 = sum([sum([math.exp(-sum([(vectormap[word][k]-c1[k])**2 if vectormap.has_key(word) else 100 for k in xrange(2)])/(2*1**2)) for c1 in vclass1])/math.sqrt(2*math.pi) for word in words_title])
	s2 = sum([sum([math.exp(-sum([(vectormap[word][k]-c2[k])**2 if vectormap.has_key(word) else 100 for k in xrange(2)])/(2*1**2)) for c2 in vclass2])/math.sqrt(2*math.pi) for word in words_title])
	s3 = sum([sum([math.exp(-sum([(vectormap[word][k]-c3[k])**2 if vectormap.has_key(word) else 100 for k in xrange(2)])/(2*1**2)) for c3 in vclass3])/math.sqrt(2*math.pi) for word in words_title])
	words_content = [item[0] for item in filter(lambda x:x[1] in ["a","an","b","n","v","vn","vi"], [item.split("/") for item in line.strip().split("\t")[1].split(" ")])]
	s4 = sum([sum([math.exp(-sum([(vectormap[word][k]-c1[k])**2 if vectormap.has_key(word) else 100 for k in xrange(2)])/(2*1**2)) for c1 in vclass1])/math.sqrt(2*math.pi) for word in words_content])
	s5 = sum([sum([math.exp(-sum([(vectormap[word][k]-c2[k])**2 if vectormap.has_key(word) else 100 for k in xrange(2)])/(2*1**2)) for c2 in vclass2])/math.sqrt(2*math.pi) for word in words_content])
	s6 = sum([sum([math.exp(-sum([(vectormap[word][k]-c3[k])**2 if vectormap.has_key(word) else 100 for k in xrange(2)])/(2*1**2)) for c3 in vclass3])/math.sqrt(2*math.pi) for word in words_content])
	X.append([float(s1*1000)/(len(words_title)+1), float(s2*1000)/(len(words_title)+1), float(s3*1000)/(len(words_title)+1), float(s4*1000)/(len(words_content)+1), float(s5*1000)/(len(words_content)+1), float(s6*1000)/(len(words_content)+1)])
	y.append(1 if 0<=i<20 else 2 if 20<=i<40 else 3 if 40<=i<60 else 0)
X, y = np.array(X), np.array(y)
# total, right = 0, 0
# clf = SVC(kernel='linear', class_weight='auto')
# clf.fit(X, y)
# for i in xrange(len(X)):
# 	total += 1
# 	if clf.predict(X[i]) == y[i]:
# 		right += 1
# 	else:
# 		print i, X[i], clf.predict(X[i])
# print float(right)/total
total, right = 0, 0
for t in xrange(10):
	print t
	kf = KFold(len(X), n_folds=10)
	for train, test in kf:
		clf = SVC(kernel='linear', class_weight='auto')
		clf.fit(X[train], y[train])
		for item in test:
			total += 1
			if clf.predict(X[item]) == y[item]:
				right += 1
print float(right)/total
# # 0.79

vectormap, vclass1, vclass2, vclass3 = {}, [], [], []
for line in fileinput.input("record.txt"):
	part = line.strip().split("\t")
	vectormap[part[2]] = [float(part[4].split(",")[0]), float(part[4].split(",")[1])]
	if part[1] == "1":
		vclass1.append([float(part[4].split(",")[0]), float(part[4].split(",")[1])])
	if part[1] == "2":
		vclass2.append([float(part[4].split(",")[0]), float(part[4].split(",")[1])])
	if part[1] == "3":
		vclass3.append([float(part[4].split(",")[0]), float(part[4].split(",")[1])])
fileinput.close()
print vclass1, vclass2, vclass3
i, X, y = 0, [], []
l1, l2, l3, l4 = [], [], [], []
for line in fileinput.input("../../data/testset/weibo_seed.txt"):
	i += 1
	words = [item[0] for item in filter(lambda x:x[1] in ["a","an","b","n","v","vn","vi"], [item.split("/") for item in line.strip().split(" ")])]
	s1 = sum([sum([math.exp(-sum([(vectormap[word][k]-c1[k])**2 if vectormap.has_key(word) else 100 for k in xrange(2)])/(2*1**2)) for c1 in vclass1])/math.sqrt(2*math.pi) for word in words])
	s2 = sum([sum([math.exp(-sum([(vectormap[word][k]-c2[k])**2 if vectormap.has_key(word) else 100 for k in xrange(2)])/(2*1**2)) for c2 in vclass2])/math.sqrt(2*math.pi) for word in words])
	s3 = sum([sum([math.exp(-sum([(vectormap[word][k]-c3[k])**2 if vectormap.has_key(word) else 100 for k in xrange(2)])/(2*1**2)) for c3 in vclass3])/math.sqrt(2*math.pi) for word in words])
	# print i, float(s1)/(len(words)+1), float(s2)/(len(words)+1), float(s3)/(len(words)+1)
	X.append([float(s1*1000)/(len(words)+1), float(s2*1000)/(len(words)+1), float(s3*1000)/(len(words)+1)])
	y.append(1 if 0<=i<20 else 2 if 20<=i<40 else 3 if 40<=i<60 else 0)
	if 0<i<=20:
		l1.append([float(s1*1000)/(len(words)+1), float(s2*1000)/(len(words)+1), float(s3*1000)/(len(words)+1)])
	elif 20<i<=40:
		l2.append([float(s1*1000)/(len(words)+1), float(s2*1000)/(len(words)+1), float(s3*1000)/(len(words)+1)])
	elif 40<i<=60:
		l3.append([float(s1*1000)/(len(words)+1), float(s2*1000)/(len(words)+1), float(s3*1000)/(len(words)+1)])
	else:
		l4.append([float(s1*1000)/(len(words)+1), float(s2*1000)/(len(words)+1), float(s3*1000)/(len(words)+1)])
	print i
X, y = np.array(X), np.array(y)
# total, right = 0, 0
# clf = SVC(kernel='linear', class_weight='auto')
# clf.fit(X, y)
# for i in xrange(len(X)):
# 	total += 1
# 	if clf.predict(X[i]) == y[i]:
# 		right += 1
# 	else:
# 		print i, X[i], clf.predict(X[i])
# print float(right)/total
total, right = 0, 0
for t in xrange(10):
	kf = KFold(len(X), n_folds=10)
	for train, test in kf:
		clf = SVC(kernel='linear', class_weight='auto')
		clf.fit(X[train], y[train])
		for item in test:
			total += 1
			if clf.predict(X[item]) == y[item]:
				right += 1
print float(right)/total
# # 0.85

# 新闻分类
vectormap, exist, vclass1, vclass2, vclass3 = {}, {}, [], [], []
for line in fileinput.input("record.txt"):
	part = line.strip().split("\t")
	vectormap[part[2]] = [float(part[4].split(",")[0]), float(part[4].split(",")[1])]
	if part[1] == "1":
		exist[part[2]] = True
		vclass1.append([float(part[4].split(",")[0]), float(part[4].split(",")[1])])
	if part[1] == "2":
		exist[part[2]] = True
		vclass2.append([float(part[4].split(",")[0]), float(part[4].split(",")[1])])
	if part[1] == "3":
		exist[part[2]] = True
		vclass3.append([float(part[4].split(",")[0]), float(part[4].split(",")[1])])
fileinput.close()
print vclass1, vclass2, vclass3
i, X, y = 0, [], []
for line in fileinput.input("../../data/testset/news_seed.txt"):
	i += 1
	print i
	words = [item[0] for item in filter(lambda x:exist.has_key(x[0]) and x[1] in ["a","an","b","n","v","vn","vi"], [item.split("/") for item in line.strip().split(" ")])]
	length = len([item[0] for item in filter(lambda x:x[1] in ["a","an","b","n","v","vn","vi"], [item.split("/") for item in line.strip().split(" ")])])
	words_title = [item[0] for item in filter(lambda x:x[1] in ["a","an","b","n","v","vn","vi"], [item.split("/") for item in line.strip().split("\t")[0].split(" ")])]
	s1 = sum([sum([math.exp(-sum([(vectormap[word][k]-c1[k])**2 if vectormap.has_key(word) else 100 for k in xrange(2)])/(2*1**2)) for c1 in vclass1])/math.sqrt(2*math.pi) for word in words_title])
	s2 = sum([sum([math.exp(-sum([(vectormap[word][k]-c2[k])**2 if vectormap.has_key(word) else 100 for k in xrange(2)])/(2*1**2)) for c2 in vclass2])/math.sqrt(2*math.pi) for word in words_title])
	s3 = sum([sum([math.exp(-sum([(vectormap[word][k]-c3[k])**2 if vectormap.has_key(word) else 100 for k in xrange(2)])/(2*1**2)) for c3 in vclass3])/math.sqrt(2*math.pi) for word in words_title])
	words_content = [item[0] for item in filter(lambda x:x[1] in ["a","an","b","n","v","vn","vi"], [item.split("/") for item in line.strip().split("\t")[1].split(" ")])]
	s4 = sum([sum([math.exp(-sum([(vectormap[word][k]-c1[k])**2 if vectormap.has_key(word) else 100 for k in xrange(2)])/(2*1**2)) for c1 in vclass1])/math.sqrt(2*math.pi) for word in words_content])
	s5 = sum([sum([math.exp(-sum([(vectormap[word][k]-c2[k])**2 if vectormap.has_key(word) else 100 for k in xrange(2)])/(2*1**2)) for c2 in vclass2])/math.sqrt(2*math.pi) for word in words_content])
	s6 = sum([sum([math.exp(-sum([(vectormap[word][k]-c3[k])**2 if vectormap.has_key(word) else 100 for k in xrange(2)])/(2*1**2)) for c3 in vclass3])/math.sqrt(2*math.pi) for word in words_content])
	X.append([float(s1*1000)/(len(words_title)+1), float(s2*1000)/(len(words_title)+1), float(s3*1000)/(len(words_title)+1), float(s4*1000)/(len(words_title)+1), float(s5*1000)/(len(words_title)+1), float(s6*1000)/(len(words_title)+1)])
	y.append(1 if 0<=i<20 else 2 if 20<=i<40 else 3 if 40<=i<60 else 0)
X, y = np.array(X), np.array(y)
clf = SVC(kernel='linear', class_weight='auto')
clf.fit(X, y)
file1 = open("../../data/classify/tag_pos_t_lable_group_comp_1.seg.txt","w")
file2 = open("../../data/classify/tag_neg_t_lable_group_comp_1.seg.txt","w")
i = 0
for line in gzip.open("../../data/original/t_lable_group_comp_1.sort.seg.txt.gz"):
	i += 1
	print i
	try:
		line.decode("utf-8")
		cid = line.strip().split("\t")[0][1:-1]
		day = (int(time.mktime(time.strptime(line.strip().split("\t")[1][1:-1],'%Y-%m-%d %H:%M:%S')))-int(time.mktime(time.strptime("2011-04-01 00:00:00",'%Y-%m-%d %H:%M:%S'))))/(24*3600)
		title = line.strip().split("\t")[4][1:-1].split(" ")
		content = line.strip().split("\t")[7][1:-1].split(" ")
		nr_map_t, ns_map_t = {}, {}
		for item in title:
			if len(item.split("/")) == 2 and item.split("/")[1] in ["nr","nr1","nr2","nrj","nrf"] and len(item.split("/")[0])/3>=2:
				nr_map_t[item.split("/")[0]] = 1 if not nr_map_t.has_key(item.split("/")[0]) else nr_map_t[item.split("/")[0]]+1
			if len(item.split("/")) == 2 and item.split("/")[1] in ["ns","nsf"] and len(item.split("/")[0])/3>=2:
				ns_map_t[item.split("/")[0]] = 1 if not ns_map_t.has_key(item.split("/")[0]) else ns_map_t[item.split("/")[0]]+1
		nr_map_c, ns_map_c = {}, {}
		for item in content:
			if len(item.split("/")) == 2 and item.split("/")[1] in ["nr","nr1","nr2","nrj","nrf"] and len(item.split("/")[0])/3>=2:
				nr_map_c[item.split("/")[0]] = 1 if not nr_map_c.has_key(item.split("/")[0]) else nr_map_c[item.split("/")[0]]+1
			if len(item.split("/")) == 2 and item.split("/")[1] in ["ns","nsf"] and len(item.split("/")[0])/3>=2:
				ns_map_c[item.split("/")[0]] = 1 if not ns_map_c.has_key(item.split("/")[0]) else ns_map_c[item.split("/")[0]]+1
		title_orig = "".join([item.split("/")[0] for item in title])
		content = content[0:500]
		length_title = len([item[0] for item in filter(lambda x:x[1] in ["a","an","b","n","v","vn","vi"], [item.split("/") for item in title])])
		words_title = [item[0] for item in filter(lambda x:exist.has_key(x[0]) and x[1] in ["a","an","b","n","v","vn","vi"], [item.split("/") for item in title])]
		s1 = sum([sum([math.exp(-sum([(vectormap[word][k]-c1[k])**2 if vectormap.has_key(word) else 100 for k in xrange(2)])/(2*1**2)) for c1 in vclass1])/math.sqrt(2*math.pi) for word in words_title])
		s2 = sum([sum([math.exp(-sum([(vectormap[word][k]-c2[k])**2 if vectormap.has_key(word) else 100 for k in xrange(2)])/(2*1**2)) for c2 in vclass2])/math.sqrt(2*math.pi) for word in words_title])
		s3 = sum([sum([math.exp(-sum([(vectormap[word][k]-c3[k])**2 if vectormap.has_key(word) else 100 for k in xrange(2)])/(2*1**2)) for c3 in vclass3])/math.sqrt(2*math.pi) for word in words_title])
		length_content = len([item[0] for item in filter(lambda x:x[1] in ["a","an","b","n","v","vn","vi"], [item.split("/") for item in content])])
		words_content = [item[0] for item in filter(lambda x:exist.has_key(x[0]) and x[1] in ["a","an","b","n","v","vn","vi"], [item.split("/") for item in content])]
		s4 = sum([sum([math.exp(-sum([(vectormap[word][k]-c1[k])**2 if vectormap.has_key(word) else 100 for k in xrange(2)])/(2*1**2)) for c1 in vclass1])/math.sqrt(2*math.pi) for word in words_content])
		s5 = sum([sum([math.exp(-sum([(vectormap[word][k]-c2[k])**2 if vectormap.has_key(word) else 100 for k in xrange(2)])/(2*1**2)) for c2 in vclass2])/math.sqrt(2*math.pi) for word in words_content])
		s6 = sum([sum([math.exp(-sum([(vectormap[word][k]-c3[k])**2 if vectormap.has_key(word) else 100 for k in xrange(2)])/(2*1**2)) for c3 in vclass3])/math.sqrt(2*math.pi) for word in words_content])
		pred = clf.predict([float(s1*1000)/(len(words_title)+1), float(s2*1000)/(len(words_title)+1), float(s3*1000)/(len(words_title)+1), float(s4*1000)/(len(words_content)+1), float(s5*1000)/(len(words_content)+1), float(s6*1000)/(len(words_content)+1)])
		if pred != 0:
			file1.write(cid+"\t"+str(pred[0])+"\t"+str(day)+"\t"+"\""+" ".join([k+":"+str(v) for k,v in nr_map_t.iteritems()])+"\""+"\t"+"\""+" ".join([k+":"+str(v) for k,v in ns_map_t.iteritems()])+"\""+"\t"+"\""+" ".join([k+":"+str(v) for k,v in nr_map_c.iteritems()])+"\""+"\t"+"\""+" ".join([k+":"+str(v) for k,v in ns_map_c.iteritems()])+"\""+"\t"+title_orig+"\n")	
		else:
			file2.write(line)
	except:
		continue
file1.close()
file2.close()

# 微博分类
vectormap, exist, vclass1, vclass2, vclass3 = {}, {}, [], [], []
for line in fileinput.input("record.txt"):
	part = line.strip().split("\t")
	vectormap[part[2]] = [float(part[4].split(",")[0]), float(part[4].split(",")[1])]
	if part[1] == "1":
		exist[part[2]] = True
		vclass1.append([float(part[4].split(",")[0]), float(part[4].split(",")[1])])
	if part[1] == "2":
		exist[part[2]] = True
		vclass2.append([float(part[4].split(",")[0]), float(part[4].split(",")[1])])
	if part[1] == "3":
		exist[part[2]] = True
		vclass3.append([float(part[4].split(",")[0]), float(part[4].split(",")[1])])
fileinput.close()
print vclass1, vclass2, vclass3
i, X, y = 0, [], []
for line in fileinput.input("../../data/testset/weibo_seed.txt"):
	i += 1
	print i
	words = [item[0] for item in filter(lambda x:exist.has_key(x[0]) and x[1] in ["a","an","b","n","v","vn","vi"], [item.split("/") for item in line.strip().split(" ")])]
	length = len([item[0] for item in filter(lambda x:x[1] in ["a","an","b","n","v","vn","vi"], [item.split("/") for item in line.strip().split(" ")])])
	s1 = sum([sum([math.exp(-sum([(vectormap[word][k]-c1[k])**2 if vectormap.has_key(word) else 100 for k in xrange(2)])/(2*1**2)) for c1 in vclass1])/math.sqrt(2*math.pi) for word in words])
	s2 = sum([sum([math.exp(-sum([(vectormap[word][k]-c2[k])**2 if vectormap.has_key(word) else 100 for k in xrange(2)])/(2*1**2)) for c2 in vclass2])/math.sqrt(2*math.pi) for word in words])
	s3 = sum([sum([math.exp(-sum([(vectormap[word][k]-c3[k])**2 if vectormap.has_key(word) else 100 for k in xrange(2)])/(2*1**2)) for c3 in vclass3])/math.sqrt(2*math.pi) for word in words])
	X.append([float(s1*1000)/(length+1), float(s2*1000)/(length+1), float(s3*1000)/(length+1)])
	y.append(1 if 0<=i<20 else 2 if 20<=i<40 else 3 if 40<=i<60 else 0)
X, y = np.array(X), np.array(y)
clf = SVC(kernel='linear', class_weight='auto')
clf.fit(X, y)
file1 = open("../../data/classify/tag_pos_t_lable_group_comp_4.seg.txt","w")
file2 = open("../../data/classify/tag_neg_t_lable_group_comp_4.seg.txt","w")
i = 0
for line in gzip.open("../../data/original/t_lable_group_comp_4.sort.seg.txt.gz"):
	i += 1
	print i
	try:
		line.decode("utf-8")
		cid = line.strip().split("\t")[0][1:-1]
		day = (int(time.mktime(time.strptime(line.strip().split("\t")[1][1:-1],'%Y-%m-%d %H:%M:%S')))-int(time.mktime(time.strptime("2011-04-01 00:00:00",'%Y-%m-%d %H:%M:%S'))))/(24*3600)
		content = line.strip().split("\t")[5][1:-1].split(" ")
		nr_map_t, ns_map_t = {}, {}
		for item in content:
			if len(item.split("/")) == 2 and item.split("/")[1] in ["nr","nr1","nr2","nrj","nrf"]:
				nr_map_t[item.split("/")[0]] = 1 if not nr_map_t.has_key(item.split("/")[0]) else nr_map_t[item.split("/")[0]]+1
			if len(item.split("/")) == 2 and item.split("/")[1] in ["ns","nsf"]:
				ns_map_t[item.split("/")[0]] = 1 if not ns_map_t.has_key(item.split("/")[0]) else ns_map_t[item.split("/")[0]]+1
		text = []
		for k,v in nr_map_t.iteritems():
			try:
				text.extend([k.decode("utf-8")]*v)
			except:
				continue
		for k,v in ns_map_t.iteritems():
			try:
				text.extend([k.decode("utf-8")]*v)
			except:
				continue
		content_orig = "".join([item.split("/")[0] for item in content])
		words = [item[0] for item in filter(lambda x:exist.has_key(x[0]) and x[1] in ["a","an","b","n","v","vn","vi"], [item.split("/") for item in content])]
		length = len([item[0] for item in filter(lambda x:x[1] in ["a","an","b","n","v","vn","vi"], [item.split("/") for item in content])])
		s1 = sum([sum([math.exp(-sum([(vectormap[word][k]-c1[k])**2 if vectormap.has_key(word) else 100 for k in xrange(2)])/(2*1**2)) for c1 in vclass1])/math.sqrt(2*math.pi) for word in words])
		s2 = sum([sum([math.exp(-sum([(vectormap[word][k]-c2[k])**2 if vectormap.has_key(word) else 100 for k in xrange(2)])/(2*1**2)) for c2 in vclass2])/math.sqrt(2*math.pi) for word in words])
		s3 = sum([sum([math.exp(-sum([(vectormap[word][k]-c3[k])**2 if vectormap.has_key(word) else 100 for k in xrange(2)])/(2*1**2)) for c3 in vclass3])/math.sqrt(2*math.pi) for word in words])
		pred = clf.predict([float(s1*1000)/(length+1), float(s2*1000)/(length+1), float(s3*1000)/(length+1)])
		if pred != 0:
			file1.write(cid+"\t"+str(pred[0])+"\t"+str(day)+"\t"+"\""+" ".join([k+":"+str(v) for k,v in nr_map_t.iteritems()])+"\""+"\t"+"\""+" ".join([k+":"+str(v) for k,v in ns_map_t.iteritems()])+"\""+"\t"+content_orig+"\n")
		else:
			file2.write(line)
	except:
		continue
file1.close()
file2.close()
