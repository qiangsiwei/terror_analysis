# Introduction

采用[知网](http://www.keenage.com/)的情感语料库,对微薄作者的发言进行情感方向分值判定.

Author: dreampuf(soddyque{aT}gmailDOTcom)

# Implement

采集 **http://v.t.sina.com.cn/widget/getusermblog.php?uid=%(WEIBOID)s**的信息并且,反序列化解析,然后对其中作者的发言部分(**i["content"]["text"] for i in ret["result"]**)进行判定.

# Improvment Plan

- 进行更多信息挖掘.
- 对判定进行平滑处理.
