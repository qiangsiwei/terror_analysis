# -*- coding: utf-8 -*-

import re
import sys
import struct
import numpy as np

#### #### #### 加载word2vec #### #### ####
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

#### #### #### 常规PCA计算 #### #### ####

from numpy import *

def loadDataSet(fileName, delim='\t'):
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    datArr = [map(float,line) for line in stringArr]
    return mat(datArr)

def pca(dataMat, topNfeat=9999999):
    meanVals = mean(dataMat, axis=0)
    meanRemoved = dataMat - meanVals #remove mean
    covMat = cov(meanRemoved, rowvar=0)
    eigVals,eigVects = linalg.eig(mat(covMat))
    eigValInd = argsort(eigVals)            #sort, sort goes smallest to largest
    eigValInd = eigValInd[:-(topNfeat+1):-1]  #cut off unwanted dimensions
    redEigVects = eigVects[:,eigValInd]       #reorganize eig vects largest to smallest
    lowDDataMat = meanRemoved * redEigVects#transform data into new dimensions
    reconMat = (lowDDataMat * redEigVects.T) + meanVals
    resultMat = np.array([reconMat[0].flatten().A[0][0:topNfeat] for i in xrange(len(reconMat))])
    return resultMat

def replaceNanWithMean(): 
    datMat = loadDataSet('secom.data', ' ')
    numFeat = shape(datMat)[1]
    for i in range(numFeat):
        meanVal = mean(datMat[nonzero(~isnan(datMat[:,i].A))[0],i]) #values that are not NaN (a number)
        datMat[nonzero(isnan(datMat[:,i].A))[0],i] = meanVal  #set NaN values to mean
    return datMat

#### #### #### tsne降维计算 #### #### ####

import numpy as Math
import pylab as Plot

def Hbeta(D = Math.array([]), beta = 1.0):
	"""Compute the perplexity and the P-row for a specific value of the precision of a Gaussian distribution."""
	# Compute P-row and corresponding perplexity
	P = Math.exp(-D.copy() * beta);
	sumP = sum(P);
	H = Math.log(sumP) + beta * Math.sum(D * P) / sumP;
	P = P / sumP;
	return H, P;
	
	
def x2p(X = Math.array([]), tol = 1e-5, perplexity = 30.0):
	"""Performs a binary search to get P-values in such a way that each conditional Gaussian has the same perplexity."""
	# Initialize some variables
	print "Computing pairwise distances..."
	(n, d) = X.shape;
	sum_X = Math.sum(Math.square(X), 1);
	D = Math.add(Math.add(-2 * Math.dot(X, X.T), sum_X).T, sum_X);
	P = Math.zeros((n, n));
	beta = Math.ones((n, 1));
	logU = Math.log(perplexity);
	# Loop over all datapoints
	for i in range(n):
		# Print progress
		if i % 500 == 0:
			print "Computing P-values for point ", i, " of ", n, "..."
		# Compute the Gaussian kernel and entropy for the current precision
		betamin = -Math.inf; 
		betamax =  Math.inf;
		Di = D[i, Math.concatenate((Math.r_[0:i], Math.r_[i+1:n]))];
		(H, thisP) = Hbeta(Di, beta[i]);
		# Evaluate whether the perplexity is within tolerance
		Hdiff = H - logU;
		tries = 0;
		while Math.abs(Hdiff) > tol and tries < 50:
			# If not, increase or decrease precision
			if Hdiff > 0:
				betamin = beta[i];
				if betamax == Math.inf or betamax == -Math.inf:
					beta[i] = beta[i] * 2;
				else:
					beta[i] = (beta[i] + betamax) / 2;
			else:
				betamax = beta[i];
				if betamin == Math.inf or betamin == -Math.inf:
					beta[i] = beta[i] / 2;
				else:
					beta[i] = (beta[i] + betamin) / 2;
			# Recompute the values
			(H, thisP) = Hbeta(Di, beta[i]);
			Hdiff = H - logU;
			tries = tries + 1;
		# Set the final row of P
		P[i, Math.concatenate((Math.r_[0:i], Math.r_[i+1:n]))] = thisP;
	# Return final P-matrix
	print "Mean value of sigma: ", Math.mean(Math.sqrt(1 / beta))
	return P;

def pca(X = Math.array([]), no_dims = 50):
	"""Runs PCA on the NxD array X in order to reduce its dimensionality to no_dims dimensions."""
	print "Preprocessing the data using PCA..."
	(n, d) = X.shape;
	X = X - Math.tile(Math.mean(X, 0), (n, 1));
	(l, M) = Math.linalg.eig(Math.dot(X.T, X));
	Y = Math.dot(X, M[:,0:no_dims]);
	return Y;

def tsne(X = Math.array([]), no_dims = 2, initial_dims = 50, perplexity = 30.0):
	"""Runs t-SNE on the dataset in the NxD array X to reduce its dimensionality to no_dims dimensions.
	The syntaxis of the function is Y = tsne.tsne(X, no_dims, perplexity), where X is an NxD NumPy array."""
	# Check inputs
	if X.dtype != "float64":
		print "Error: array X should have type float64.";
		return -1;
	# Initialize variables
	X = pca(X, initial_dims);
	(n, d) = X.shape;
	max_iter = 500;
	initial_momentum = 0.5;
	final_momentum = 0.8;
	eta = 500;
	min_gain = 0.01;
	Y = Math.random.randn(n, no_dims);
	dY = Math.zeros((n, no_dims));
	iY = Math.zeros((n, no_dims));
	gains = Math.ones((n, no_dims));
	# Compute P-values
	P = x2p(X, 1e-5, perplexity);
	P = P + Math.transpose(P);
	P = P / Math.sum(P);
	P = P * 4;# early exaggeration
	P = Math.maximum(P, 1e-12);
	# Run iterations
	for iter in range(max_iter):
		# Compute pairwise affinities
		sum_Y = Math.sum(Math.square(Y), 1);		
		num = 1 / (1 + Math.add(Math.add(-2 * Math.dot(Y, Y.T), sum_Y).T, sum_Y));
		num[range(n), range(n)] = 0;
		Q = num / Math.sum(num);
		Q = Math.maximum(Q, 1e-12);
		# Compute gradient
		PQ = P - Q;
		for i in range(n):
			dY[i,:] = Math.sum(Math.tile(PQ[:,i] * num[:,i], (no_dims, 1)).T * (Y[i,:] - Y), 0);
		# Perform the update
		if iter < 20:
			momentum = initial_momentum
		else:
			momentum = final_momentum
		gains = (gains + 0.2) * ((dY > 0) != (iY > 0)) + (gains * 0.8) * ((dY > 0) == (iY > 0));
		gains[gains < min_gain] = min_gain;
		iY = momentum * iY - eta * (gains * dY);
		Y = Y + iY;
		Y = Y - Math.tile(Math.mean(Y, 0), (n, 1));
		# Compute current value of cost function
		if (iter + 1) % 10 == 0:
			C = Math.sum(P * Math.log(P / Q));
			print "Iteration ", (iter + 1), ": error is ", C
		# Stop lying about P-values
		if iter == 100:
			P = P / 4;
	# Return solution
	return Y;

if __name__ == "__main__":
	import random
	import fileinput
	import matplotlib.pyplot as plt
	# from gensim import corpora, models, similarities

	# # 公交/b 爆炸/vn 暴力/n 恐怖/a 校园/n 砍/v
	# list1 = {"车":1,"火":1,"纵火":4,"大火":4,"起火":4,"着火":4,"扑火":1,"救火":1,"火势":1,"火光":1,"烧毁":1,"公交":5,"巴士":4,"司机":4,"乘客":4,"消防":1,"扑灭":1,"燃烧":1,"爆炸":1,"疏散":1,"面包车":1,"救护车":1,"消防车":1,"车站":1,"车辆":1,"道路":1,"马路":1}
	# list2 = {"暴力":3,"恐怖":3,"袭击":3,"暴徒":2,"犯罪":1,"抢劫":1,"爆炸":1,"炸弹":1,"枪杀":1,"缴获":1,"歹徒":1,"逃犯":1,"人质":1,"武警":1,"警方":1,"民警":1,"特警":1,"击毙":1,"嫌疑人":1,"杀人案":1,"打砸抢":1}
	# list3 = {"刀":3,"砍":3,"刺":1,"割":1,"菜刀":2,"刺伤":1,"学生":2,"孩子":1,"老师":1,"教师":1,"家长":1,"校方":1,"小学":2,"中学":2,"初中":2,"高中":2,"大学":2,"同学":1,"学校":3,"教室":1,"上课":1,"校医":1,"上学":1,"放学":1,"小学生":1,"中学生":1,"初中生":1,"高中生":1,"大学生":1}
	# vectormap = {}
	# for line in fileinput.input("../../data/word2vec/vectors.txt"):
	# 	try:
	# 		word, vector = line.strip().split("\t")[0], [float(i) for i in line.strip().split("\t")[1].split(" ")]
	# 		vectormap[word] = vector
	# 	except:
	# 		continue
	# fileinput.close()
	# wmap = {}
	# for word, value in list1.iteritems():
	# 	wmap[word] = {"vector":vectormap[word],"labels":{0:0,1:value,2:0,3:0}}
	# for word, value in list2.iteritems():
	# 	wmap[word] = {"vector":vectormap[word],"labels":{0:0,1:0,2:value,3:0}}
	# for word, value in list3.iteritems():
	# 	wmap[word] = {"vector":vectormap[word],"labels":{0:0,1:0,2:0,3:value}}
	# #### #### #### 抽取N个无关词汇 step1 #### ####
	# for i in random.sample([i for i in range(1,len(vectormap.keys()))], 500):
	# 	word = vectormap.keys()[i]
	# 	try:
	# 		if all([uchar >= u'\u4e00' and uchar<=u'\u9fa5' for uchar in word.decode("utf-8")]) and len(word.decode("utf-8")) >= 2:
	# 			print word
	# 			if not wmap.has_key(word):
	# 				wmap[word] = {"vector":vectormap[word],"labels":{0:0,1:0,2:0,3:0}}
	# 			wmap[word]["labels"][0] += 1
	# 	except:
	# 		continue
	# #### #### #### 抽取N个无关词汇 step2 #### ####
	# for line in fileinput.input("sample1.txt"):
	# 	try:
	# 		word = line.strip()
	# 		if not wmap.has_key(word):
	# 			wmap[word] = {"vector":vectormap[word],"labels":{0:0,1:0,2:0,3:0}}
	# 		wmap[word]["labels"][0] += 1
	# 	except:
	# 		print word
	# fileinput.close()
	# #### #### #### #### #### #### #### #### ####
	# X, wmapkeys = [], sorted(wmap.keys())
	# for i in xrange(len(wmapkeys)):
	# 	X.append(wmap[wmapkeys[i]]["vector"])
	# X = np.array(X)
	# Y = tsne(X, 2, 100, 20.0)
	# with open("record.txt","w") as f:
	# 	for p in xrange(4):
	# 		tops = []
	# 		for i in xrange(len(wmapkeys)):
	# 			weight = wmap[wmapkeys[i]]["labels"][p]
	# 			if weight!=0:
	# 				tops.append({"word":wmapkeys[i],"value":weight,"vector":Y[i]})
	# 		tops = sorted(tops, key=lambda x:x["value"], reverse=True)
	# 		for top in tops:
	# 			f.write("1\t"+str(p)+"\t"+top["word"]+"\t"+str(top["value"])+"\t"+",".join([str(f) for f in top["vector"]])+"\n")
	# # beta = np.linalg.lstsq(X, Y)[0]
	# # for k,v in vectormap.iteritems():
	# # 	if not wmap.has_key(k):
	# # 		file.write("0\t0\t"+k+"\t0\t"+",".join([str(f) for f in np.dot(beta.T, np.array(v))])+"\n")

	# 高维语境2维显示
	x1, y1, s1, c1 = [], [], [], []
	for line in fileinput.input("record.txt"):
		part = line.strip().split("\t")
		if part[0] == "1" and part[1] == "0":
			x1.append(float(part[4].split(",")[0]))
			y1.append(float(part[4].split(",")[1]))
			s1.append(int(part[3])*25)
			c1.append("k")
		if part[0] == "1" and part[1] == "1":
			x1.append(float(part[4].split(",")[0]))
			y1.append(float(part[4].split(",")[1]))
			s1.append(int(part[3])*25)
			c1.append("r")
		if part[0] == "1" and part[1] == "2":
			x1.append(float(part[4].split(",")[0]))
			y1.append(float(part[4].split(",")[1]))
			s1.append(int(part[3])*25)
			c1.append("g")
		if part[0] == "1" and part[1] == "3":
			x1.append(float(part[4].split(",")[0]))
			y1.append(float(part[4].split(",")[1]))
			s1.append(int(part[3])*25)
			c1.append("b")
	plt.figure(1)
	plt.subplot(111)
	plt.xlim(-80,80)
	plt.ylim(-80,80)
	plt.scatter(x1, y1, s=s1, color=c1, alpha=0.3)
	plt.show()

	# # 特征词人工筛选
	# for line in fileinput.input("record.txt"):
	# 	part = line.strip().split("\t")
	# 	if -20<=float(part[4].split(",")[0])<=50 and -50<=float(part[4].split(",")[1])<=50 and int(part[1])==2:
	# 		print line.strip()
	# fileinput.close()
