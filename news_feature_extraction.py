# -*- coding: utf-8 -*-

import fileinput
from gensim import corpora, models, similarities

# 独立事件特征抽取
event_map = {}
for line in fileinput.input("data/events/classified_news.txt"):
	part = line.strip().split("\t")
	event, cls, day = int(part[0]), int(part[1]), int(part[2])
	nr_t = None if len(part[5].split("|")[0][1:-1]) == 0 else [(item.split(":")[0], int(item.split(":")[-1])) for item in part[5].split("|")[0][1:-1].split(" ")]
	ns_t = None if len(part[5].split("|")[1][1:-1]) == 0 else [(item.split(":")[0], int(item.split(":")[-1])) for item in part[5].split("|")[1][1:-1].split(" ")]
	nr_c = None if len(part[5].split("|")[2][1:-1]) == 0 else [(item.split(":")[0], int(item.split(":")[-1])) for item in part[5].split("|")[2][1:-1].split(" ")]
	ns_c = None if len(part[5].split("|")[3][1:-1]) == 0 else [(item.split(":")[0], int(item.split(":")[-1])) for item in part[5].split("|")[3][1:-1].split(" ")]
	text, weight = [], 5
	if not event_map.has_key(event):
		event_map[event] = {"cls":cls,"stime":day,"nr_t":{},"ns_t":{},"nr_c":{},"ns_c":{}}
	if nr_t != None:
		for w in nr_t:
			event_map[event]["nr_t"][w[0]] = w[1] if not event_map[event]["nr_t"].has_key(w[0]) else event_map[event]["nr_t"][w[0]]+w[1]
	if ns_t != None:
		for w in ns_t:
			event_map[event]["ns_t"][w[0]] = w[1] if not event_map[event]["ns_t"].has_key(w[0]) else event_map[event]["ns_t"][w[0]]+w[1]
	if nr_c != None:
		for w in nr_c:
			event_map[event]["nr_c"][w[0]] = w[1] if not event_map[event]["nr_c"].has_key(w[0]) else event_map[event]["nr_c"][w[0]]+w[1]
	if ns_c != None:
		for w in ns_c:
			event_map[event]["ns_c"][w[0]] = w[1] if not event_map[event]["ns_c"].has_key(w[0]) else event_map[event]["ns_c"][w[0]]+w[1]
fileinput.close()
with open("data/events/classified_event.txt","w") as f:
	for k,v in event_map.iteritems():
		nr_t = " ".join([item["a"]+":"+str(item["b"]) for item in sorted([{"a":a,"b":b} for a,b in v["nr_t"].iteritems()], key=lambda x:x["b"], reverse=True)])
		ns_t = " ".join([item["a"]+":"+str(item["b"]) for item in sorted([{"a":a,"b":b} for a,b in v["ns_t"].iteritems()], key=lambda x:x["b"], reverse=True)])
		nr_c = " ".join([item["a"]+":"+str(item["b"]) for item in sorted([{"a":a,"b":b} for a,b in v["nr_c"].iteritems()], key=lambda x:x["b"], reverse=True)])
		ns_c = " ".join([item["a"]+":"+str(item["b"]) for item in sorted([{"a":a,"b":b} for a,b in v["ns_c"].iteritems()], key=lambda x:x["b"], reverse=True)])
		f.write(str(k)+"|"+str(v["cls"])+"|"+str(v["stime"])+"|"+nr_t+"|"+ns_t+"|"+nr_c+"|"+ns_c+"\n")

# 事件对应
def cos(a, b):
	import math
	try:
		m, l = {}, []
		for (pa,sa) in a:
			m[pa] = sa
		for (pb,sb) in b:
			if m.has_key(pb):
				l.append([m[pb],sb])
		return sum([i[0]*i[1] for i in l])/math.sqrt(sum([sa**2 for (pa,sa) in a])*sum([sb**2 for (pb,sb) in b]))
	except:
		return 0

weight, events, corpus_tfidf = 3, [], []
for line in fileinput.input("data/events/classified_feat.txt"):
	events.append([int(line.replace("\n","").split("\t")[1]),-1])
	corpus_tfidf.append([(int(item.split(":")[0]), float(item.split(":")[1])) for item in filter(lambda x:x!='', line.replace("\n","").split("\t")[-1].split(" "))])
fileinput.close()
eventclass, mino, eid = {1:[],2:[],3:[]}, 0.5, 0
for i in xrange(len(corpus_tfidf)):
	print i
	maxsim, assign, ne, feature = 0, -1, -1, corpus_tfidf[i]
	for e in xrange(len(eventclass[events[i][0]])):
		simo = cos(eventclass[events[i][0]][e]["feature"],feature)
		if simo >= mino and simo > maxsim:
			maxsim, assign, ne = simo, eventclass[events[i][0]][e]["eid"], e
	if assign != -1:
		events[i][1], fmap = assign, {}
		for (p,s) in eventclass[events[i][0]][ne]["feature"]:
			fmap[p] = s
		for (p,s) in feature:
			fmap[p] = s if not fmap.has_key(p) else fmap[p]+s
		eventclass[events[i][0]][ne]["feature"] = [(p,s) for p,s in fmap.iteritems()]
	else:
		events[i][1] = eid
		eventclass[events[i][0]].append({"eid":eid,"feature":feature})
		eid += 1
with open("data/events/classified_mapping.txt","w") as f:
	for i in xrange(len(events)):
		f.write(str(events[i][0])+"\t"+str(events[i][1])+"\n")
