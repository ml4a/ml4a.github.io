---
layout: chapter
title: "Language processing"
---


Quote

"the trophy doesn't fit in the suitcase because it is too big" - geoff hinton
"the trophy doesn't fit in the suitcase because it is too small" - geoff hinton

Trying to get algorithms to make sense of ambiguity of human language, we begin to appreciate just how much we take for granted. We hardly notice the tiny feats of disambiguation our brains do when reading sentences like the ones above.

you shall know the nature of a word by the company it keeps

----

What kinds of language-oriented tasks might we be interested in?

----

This chapter is about applications of machine learning to natural language processing. like ml, NLP is a nebulous term with several precise definitions and most have something to do wth making sense from text. This chapter will take a broad view of NLP


“Deep Learning waves have lapped at the shores of computational linguistics for several years now, but 2015 seems like the year when the full force of the tsunami hit the major Natural Language Processing (NLP) conferences.” -Dr. Christopher D. Manning, Dec 2015 

 - check manning pdf statement

quote/link from: http://www.andreykurenkov.com/writing/a-brief-history-of-neural-nets-and-deep-learning/


word vectors

word vectors are a rep such that geometric preserved in emeddings.
reverse king queen

cover tf-idf in detail (link to it fromm  tsne chapter). link to t-sne chapter from here

since then, there have been a number of writings which have tried to interpret these word vectors. gender binary


tf-idf examples -> t-SNE examples

aparrish generative poetry

word2vec
 - analogies
 - kcimc antonyms
 - rejecting gender binary


tf-idf -> t-SNE
LSA + LDA -> t-SNE

RNNs annotating?


word2vec chapter
 - anything2vec http://www.lab41.org/anything2vec/

captioning
 - NeuralTalk and Walk

Mario RNN

attention + DRAW


https://www.youtube.com/watch?v=XG-dwZMc7Ng
the trophy can't fit into the suitcase because it's too big (it = trophy)
the trophy can't fit into the suitcase because it's too small (it = suitcase)



colah word2vec http://colah.github.io/posts/2014-07-NLP-RNNs-Representations/

metal + NLP
https://www.reddit.com/r/MachineLearning/comments/4r1np7/heavy_metal_and_natural_language_processing_part_1/?utm_source=twitterfeed&utm_medium=twitter

https://github.com/explosion/spaCy/tree/master/examples/keras_parikh_entailment

http://sebastianruder.com/secret-word2vec/index.html

https://civisanalytics.com/blog/data-science/2016/09/22/neural-network-visualization/


lda2vec Chris Moody hybrid word2vec and LDA  http://multithreaded.stitchfix.com/blog/2016/05/27/lda2vec/#topic=38&lambda=1&term=

historical word embeddings http://nlp.stanford.edu/projects/histwords/

textsum https://github.com/tensorflow/models/tree/master/textsum

http://wiki.dbpedia.org/Datasets/NLP%20https://datahub.io/dataset?tags=nlp

doc2vec http://nbviewer.jupyter.org/github/fbkarsdorp/doc2vec/blob/master/doc2vec.ipynb

Language modeling a billion words http://torch.ch/blog/2016/07/25/nce.html

demystifying word2vec https://buss_jan.gitbooks.io/word2vec/content/chapter2.html
https://github.com/facebookresearch/fastText

keras word2vec https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/word2vec/word2vec_basic.py

hotel reviews: https://blog.monkeylearn.com/machine-learning-1m-hotel-reviews-finds-interesting-insights/

https://github.com/thoppe/transorthogonal-linguistics
rejecting gender binary http://bookworm.benschmidt.org/posts/2015-10-30-rejecting-the-gender-binary.html

Voynich Manuscript: word vectors and t-SNE visualization of some patterns blog.christianperone.com/2016/01/voynich-manuscript-word-vectors-and-t-sne-visualization-of-some-patterns/

kcimc synonyms + antonyms
https://gist.github.com/kylemcdonald/3463caf86ffca5c950c2
https://gist.github.com/kylemcdonald/3463caf86ffca5c950c2
https://gist.github.com/kylemcdonald/9bedafead69145875b8c#file-_tsne-pdf

CNN sentence classificaiton https://github.com/yoonkim/CNN_sentence

Chris Olah Word2Vec + tSNE http://colah.github.io/posts/2014-07-NLP-RNNs-Representations/

paragraph vectors: https://arxiv.org/pdf/1507.07998.pdf

text_analytics_on_mpp doc2vec from newsgroups https://github.com/vatsan/text_analytics_on_mpp/blob/master/neural_language_models/01_news_groups_doc2vec.ipynb

Harvard NLP https://github.com/harvardnlp
Stanford NLP

https://lamyiowce.github.io/word2viz/



===========


Datasets
- Common schema for datasets
- dowloading images from google [python] [js]
- Freely available datasets
- Extracting/scraping data from the web
- Data "mungling"

Feature extraction and word embeddings
 - Bag of words
 - tf-idf
 - latent dirichlet, lsa
 - word2vec, doc2vec

Organizing, retrieving documents
 - document classification
 - clustering and visualizing documents
 - document retrieval, similarity ranking
 - combining with filters

NLP tasks
 - sentiment analysis
 - named entity recognition
 - quote attribution
 - anomaly detection


Speculative NLP tasks
 - fact-checking (https://fullfact.org/blog/2016/aug/automated-factchecking/)

*    - topic-modeling (tfidf, lsa, lda)
*    - document retrieval/similarity/visualization
*    - word2vec
*    - sentiment analysis
*    - skip-thoughts/doc2vec/lda2vec


 https://www.quora.com/What-are-good-resources-tutorials-to-learn-Keras-deep-learning-library-in-Python
 http://u.cs.biu.ac.il/~yogo/nnlp.pdf
 http://rare-technologies.com/making-sense-of-word2vec/
 http://lxmls.it.pt/2014/socher-lxmls.pdf
 http://nlp.stanford.edu/courses/NAACL2013/NAACL2013-Socher-Manning-DeepLearning.pdf
 http://nlp.stanford.edu/~socherr/SocherBengioManning-DeepLearning-ACL2012-20120707-NoMargin.pdf
 https://github.com/jtoy/awesome-tensorflow/

 db
 http://www-nlp.stanford.edu/links/statnlp.html
 https://datahub.io/dataset?tags=nlp
 http://wiki.dbpedia.org/Datasets/NLP
 extract wikipedia https://github.com/bwbaugh/wikipedia-extractor


links
- debiasing embeddings http://arxiv.org/pdf/1607.06520.pdf
- https://github.com/facebookresearch/fastText
- http://nlp.stanford.edu/projects/glove/   
NLP topics
- semantic hashing for fast document retrieval (use auto encoder to learn binary addresses for documents, then use it as a memory address for a hash map and look for documents in nearby memory cells — very fast

ideas
- skipgram retrieval


syntaxNet
https://research.googleblog.com/2016/05/announcing-syntaxnet-worlds-most.html


Syllabus
======

dimensionality reduction
- explanation of manifolds
- PCA + SVD
- eigenfaces

text representations
- tf-idf + bag-of-words
- lsa/lda

applications of text representations
- document retrieval/similarity
- document clustering/visualization
- topic modeling

word vectors

paragraph vectors / skip-thoughts
- nearest skip-thought retrieval
- next skip-thought prediction

NLP tasks
- named entity recognition
- POS tagging
- sentiment analysis
- translation
- summarization
- 
speculative nlp tasks
- stylometry / deanonymization

etc
- semantic hashing + fast retrieval
- summarization TextSum
 
NOTEBOOKS
- document retrieval
    - tf-idf
    - lsa/lda
- document clustering, organization, visualization
    - unsupervised: t-SNE
    - topic modeling
    - classification: neural net
- word2vec + t-SNE
- skip-thought vectors

SOFTWARE
- gensim, sklearn

https://github.com/explosion/spaCy/tree/master/examples/keras_parikh_entailment

sebastian ruder blog: http://sebastianruder.com/word-embeddings-softmax/index.html#hierarchicalsoftmax

https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md