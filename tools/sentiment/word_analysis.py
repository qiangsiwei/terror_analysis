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
	from gensim import corpora, models, similarities

	# print "Run Y = tsne.tsne(X, no_dims, perplexity) to perform t-SNE on your dataset."
	# print "Running example on 2,500 MNIST digits..."
	# X = Math.loadtxt("mnist2500_X.txt");
	# labels = Math.loadtxt("mnist2500_labels.txt");
	# X, labels = X[0:100], labels[0:100]
	# Y = tsne(X, 2, 50, 20.0);
	# Plot.scatter(Y[:,0], Y[:,1], 20, labels);
	# plt.scatter(Y[:,0], Y[:,1], 20, labels)
	# plt.show()

	vectormap = {}
	for line in fileinput.input("../../data/word2vec/vectors.weibo.txt"):
		try:
			word, vector = line.strip().split("\t")[0], [float(i) for i in line.strip().split("\t")[1].split(" ")]
			vectormap[word] = vector
		except:
			continue
	fileinput.close()
	W, X, y = [], [], []
	for line in fileinput.input("data/pos_eva.sort.txt"):
		word = line.strip()
		if vectormap.has_key(word) and len(word.decode("utf-8"))>=2:
			W.append(word)
			X.append(vectormap[word])
			y.append("1")
			if len(W) == 200:
				break
	fileinput.close()
	for line in fileinput.input("data/pos_emo.sort.txt"):
		word = line.strip()
		if vectormap.has_key(word) and len(word.decode("utf-8"))>=2:
			W.append(word)
			X.append(vectormap[word])
			y.append("2")
			if len(W) == 400:
				break
	fileinput.close()
	for line in fileinput.input("data/neg_eva.sort.txt"):
		word = line.strip()
		if vectormap.has_key(word) and len(word.decode("utf-8"))>=2:
			W.append(word)
			X.append(vectormap[word])
			y.append("3")
			if len(W) == 600:
				break
	fileinput.close()
	for line in fileinput.input("data/neg_emo.sort.txt"):
		word = line.strip()
		if vectormap.has_key(word) and len(word.decode("utf-8"))>=2:
			W.append(word)
			X.append(vectormap[word])
			y.append("4")
			if len(W) == 800:
				break
	fileinput.close()
	for line in fileinput.input("data/newword.txt"):
		word = line.strip()
		if vectormap.has_key(word) and len(word.decode("utf-8"))>=2:
			W.append(word)
			X.append(vectormap[word])
			y.append("5")
			if len(W) == 200:
				break
	fileinput.close()
	#### #### #### 抽取N个无关词汇 #### #### ####
	# for i in random.sample([i for i in range(1,len(vectormap.keys()))], 1600):
	# 	word = vectormap.keys()[i]
	# 	try:
	# 		if all([uchar >= u'\u4e00' and uchar<=u'\u9fa5' for uchar in word.decode("utf-8")]):
	# 			W.append(word)
	# 			X.append(vectormap[word])
	# 			y.append("0")
	# 	except:
	# 		continue
	#### #### #### ############ #### #### ####
	X, y = np.array(X), np.array(y)
	Y = tsne(X, 2, 100, 20.0)
	with open("record.txt","w") as f:
		for i in xrange(len(W)):
			f.write(W[i]+"\t"+y[i]+"\t"+str(Y[i][0])+","+str(Y[i][1])+"\n")

	# 高维语境2维显示
	x0, y0, s0, c0 = [], [], [], []
	for line in fileinput.input("record.txt"):
		part = line.strip().split("\t")
		if part[1] == "1":
			x0.append(float(part[2].split(",")[0]))
			y0.append(float(part[2].split(",")[1]))
			c0.append("r")
		if part[1] == "2":
			x0.append(float(part[2].split(",")[0]))
			y0.append(float(part[2].split(",")[1]))
			c0.append("g")
		if part[1] == "3":
			x0.append(float(part[2].split(",")[0]))
			y0.append(float(part[2].split(",")[1]))
			c0.append("b")
		if part[1] == "4":
			x0.append(float(part[2].split(",")[0]))
			y0.append(float(part[2].split(",")[1]))
			c0.append("y")
		if part[1] == "0":
			x0.append(float(part[2].split(",")[0]))
			y0.append(float(part[2].split(",")[1]))
			c0.append("k")
	plt.figure(1)
	plt.subplot(111)
	plt.xlim(-100,100)
	plt.ylim(-100,100)
	plt.scatter(x0, y0, color=c0, alpha=0.3)
	plt.show()

	# # 特征词人工筛选
	# for line in fileinput.input("record.txt"):
	# 	part = line.strip().split("\t")
	# 	if -100<=float(part[4].split(",")[0])<=-45 and -50<=float(part[4].split(",")[1])<=-30:
	# 		print line.strip()
	# fileinput.close()

	# # 构建无关词表
	# outliers = []
	# for line in fileinput.input("outlier.txt"):
	# 	outliers.append(line.strip())
	# fileinput.close()
	# with open("outlier.txt","w") as f:
	# 	f.write("[\""+"\",\"".join(outliers)+"\"]")
