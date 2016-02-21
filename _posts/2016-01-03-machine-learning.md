---
layout: post
title: "Machine learning"
date: 2016-01-03
---

top link
 - banner image with many examples of ML art
 - more images
 - tweet images

multilayer NN are universal approximators
 - http://www.sciencedirect.com/science/article/pii/0893608089900208

http://frnsys.com/ai_notes/foundations/linear_algebra.html



You\'ve heard by now that machine learning refers to a broad set of techniques which allow computers to learn from data. But learn what exactly, and how? To understand this, we\'ll use several concrete examples in which techniques from machine learning can be applied.

**Example 1**: Suppose you are a climatologist who is trying to devise a computer program which can predict whether or not it will rain on a given day. Turns out this is hard! But we understand intuitively that rain has something to do with the temperature, atmospheric pressure, humidity, wind, cloud cover, location and time of year, and so on.

**Example 2**: Gmail, Yahoo, and other e-mail services provide tools to automatically filter out spam e-mails before they reach your inbox. Like in the rain example, we have a few intuitions about this task. E-mails containing phrases like \"make $$$ now\" or \"free weight loss pills\" are probably suspicious, and we can surely think up a few more. Of course the presence of one suspicious term does not guarantee it\'s spam, and so we can\'t take the naive approach of labeling spam for any e-mail containing any suspicious phrase.

One way to approach solving these problems is a \"rule-based\" or \"expert\" approach in which a series of rules are carefully designed and tested at runtime to determine the output. In the spam example, this could take the form of a [decision tree](___). Upon receiving an e-mail, check to see if it\'s from an unknown sender; if it is, check to see if the phrase \"lose weight now!\" appears, and if it does appear and there is a link to an unknown website, classify it as spam. Obviously our decision tree would be much larger and more complicated than this, but it would still be characterized by a sequence of if-then statements leading to a decision.

Such a strategy suffers from two major weaknesses. First, it requires a great deal of expert guidance and hand-engineering which may be time-consuming and costly. Furthermore, spam trigger words and global climate patterns change continuously, and we\'d have to reprogram them every so often for them to remain effective. Secondly, a rule-based approach does not _generalize_. Notice our spam decision tree won\'t help us predict the rain, or vice-versa, nor will they easily apply to other problems we haven\'t talked about. Expert systems like these are domain-specific, and if our task changes even slightly, our carefully crafted algorithm must be reconstructed from scratch.

# Learning from past observations

With machine learning, we take a different approach. We start by reducing these two very different example problems given above to essentially the same generic task: given a set of observations about something, make a decision, or _**classification**_. Rain or no rain; spam or not spam. In other problem domains we may have more than 2 choices. Or we may have one continuous value to predict, e.g. _how much_ it will rain. In this last case, we call this problem _**regression**_. In each of these cases, we have posed a single abstract problem: determine the relationship between our observations or data, and our desired task.

[Known observations] -> [Learning] <- [Known outputs]
                            ||
[Unknown observation] -> [ Model ] -> Predicted output

Furthermore, machine learning also takes the position that such a functional relationship can be _learned_ from past observations and their known outputs. For the rain prediction problem, we may have a database with thousands of examples where those variables we think are important (pressure, temperature, etc) were measured and we know whether or not it actually rained those days. In the spam example, we may have a database of e-mails which were labeled as spam or not spam by a human. Using this data, we can craft a function which is able to modify its own internal structure in response to new observations, so as to be able to improve its ability to accurately perform the task. Formally, the set of previous examples with its known outcomes is often called a _ground truth_ and it is used as a _training set_ to _train_ our predictive algorithm. 

More generally, what\'s been defined in this section is called _**supervised learning**_ and is one of the foundational branches of machine learning. _**Unsupervised learning**_ refers to tasks involving data which is unlabeled, and _**reinforcement learning**_ is a hybrid of the two, but we will get to those later.

# The simplest machine learning algorithm: a linear classifier

We've introduced the notion of an algorithm which which makes a series of empirical observations about something and uses those observations to make a decision about something  data-driven machine learning

Using that as our blueprint, we will now make our first predictive model, a simple linear classifier. A linear classifier is defined as a function of our data, $$X$$, and 



[ 1 ] - 2d line classifier
[ 2 ] - 3d plane classifier

sometimes our data is not linearly separable. take the simple example of __
[ 3 ] - 2d non-linearly separable 

# Dimension X


matplotlib: http://matplotlib.org/examples/mplot3d/surface3d_demo.html
do gif rotation of planar classifier

A flat plane in 3d is analogous to a line in 2d, and is thus called \"linear.\" This is true in general for any n-dimensional hyperplane. Linear classifiers are limited because in reality, most problems that interest us don't have such flat behavior; different variables interact in various ways.


# no labels

Later, unsupervised

# Connecting the dots

It may seem hard to believe, but this simple setup forms the core of 

the same linear classifier which is simply telling apart two objects from each other are the thing that underpins puppy slugs, shakespeare bots, word embeddings (word2vec), and others. this may seem hard to believe at first but it rests on a subtle point. When we train an algorithm to discern the relationship between a set of observations and a corresponding behavior, we are doing more than letting it make new predictions. We are also forming a _representation_ of our subject of interest, a computational schematic



** taken together, the weights and the biases are often also called _parameters_ because the behavior of our machine depends on how we set them. 





looking inside neural nets
 - top bar: snow angel CIFAR blobs

how neural nets are trained
 - top bar: mountains

deepdream
 - top bar: mike tyka's original recursive ones
 - on june __ someone mysteriously posted a photo
 - mike tyka's new deepdream experiments

convnets
 - top bar: ofxCcv viewer animated gif of me waving

style transfer
 - top bar: multiple mona lisas 

Wekinator and real-time performance with neural nets
 - quote from rebecca
 - top bar: some application (phoenix?)

self-organizing maps
 - top bar: flower SOMs or color of words

t-SNE
 - top bar: grid of flowers/animals, or karpathy's grid?
 - include olivia jack, moritz stefaner, myself, golan + aman

NLP
 - top bar: kyle's ragas or my wikipedia concepts
 - quote: chris manning, deep learning will steamroll NLP
 - word2vec
 - translation feedback loops?

ethics
 - top bar: heather's faces
 - kate crawford, hanna wallach
 - many art projects offer a glimpse into a brave new world, filled with 
 - corporations prefer deep learning because it automates feature extraction