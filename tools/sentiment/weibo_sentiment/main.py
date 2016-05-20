#!/usr/bin/env python2.5
# -*- coding: utf-8 -*-

from __future__ import with_statement
import sys
import urllib2
from contextlib import closing

try:
    import json
except ImportError:
    try:
        import simplejson as json
    except ImportError:
        from django.utils import simplejson as json

with open("positive.txt") as f:
    po_corpus = [i.decode("u8") for i in f.xreadlines()]

with open("negative.txt") as f:
    ne_corpus = [i.decode("u8") for i in f.xreadlines()]

def main():
    usr_txt = "混战中，闵拥军的石头又被打掉，王仁明和邹老师立即扑上前将对方按住，并用绳子把闵拥军绑了起来。闵正安也提到，去年冬天快过年时，闵拥军和卖太阳能的老板产生矛盾，对方找上门来，闵拥军把老板的头打破了。".decode("utf-8")
    po = ne = 0
    po = sum(map(lambda x: len(filter(lambda word: word in x, po_corpus)), usr_txt))
    ne = sum(map(lambda x: len(filter(lambda word: word in x, ne_corpus)), usr_txt))
    print "Positive:%s\nNegative:%s" % (po, ne)

if __name__ == "__main__":
    main()

