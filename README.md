Public Safety Incident Automatic Detection
=============

This is the code repository for paper "Automatic Detection Public Safety Incident of Different Categories from Online News".

Recent increase in public safety incidents around China and all over the world have made analyzing and understanding such activities more critical than ever before. Conventionally, public safety incidents databases are constructed based on data collected from authorized organization, and maintained manually. However, this can be laborious and time-consuming, and due to the data source limitation, many local incidents tend to be neglected, which may lead to impairment in the consequent analysis. Fortunately, the ever-growing amount of on-line information can offer great opportunities for detecting and extracting incidents of different categories from online news.

Here, we propose a framework and methods to detect and public safety incidents of various categories and extract their descriptions automatically, which can be applied to a stream of news articles gathered previously or continuously. The system framework is as follows.

System framework
----

![Alt Text](https://raw.githubusercontent.com/qiangsiwei/terror_analysis/master/figure/framework.png)

News articles are firstly classified into different categories by a news classifier, based on a context-based method. Afterwards, articles are partitioned into clusters by a event detector based on text and temporal similarities. Later, key information such as the number of injury or casualty is extracted from the text based on Conditional Random Field (CRF). Our ultimate goal is to construct Global Terrorism Database fully automatically.

Our proposed context-based method for news classifier works as follows. Usually, similarities between text are calculated based on word frequency or inverse document frequency, and words are recognized as isolated such as "one hot", therefore semantic contexts are lost. However, word embeddings can map words into low-dimensional space while retain the relative relations between words, and the distance between vectors can be calculated by cosine similarity or Euclidean distance.

Our method is similar to the following work, "Gaussian LDA for Topic Models with Word Embeddings".   
<http://rajarshd.github.io/papers/acl2015.pdf>

The low dimension visualization of word similarities in high dimensional space as achieved by TSNE.

![Alt Text](https://raw.githubusercontent.com/qiangsiwei/terror_analysis/master/figure/tsne.png)
