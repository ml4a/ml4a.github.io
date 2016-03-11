---
layout: post
title: "Looking inside neural nets"
date: 2016-01-05
---

In the [previous chapter](), we saw how a neural network can be trained to classify handwritten digits with a respectable accuracy of 90% (?). In this chapter, we are going to look at its performance a little more carefully, as well as examine its internal state to develop a few intuitions about what\'s really going on. Later in this chapter, we are going to break our neural network altogether by attempting to train it on a more complicated dataset of objects like dogs, automobiles, and ships, to try to anticipate what kind of innovations will be necessary. (struggle)

 
# The template interpretation

[printblock analogy?] [borges quote about numbers]


Recall that when we input an image into our neural net, we \"unroll\" the pixels into a single row, as shown below. Rather than unrolling the pixels though, let\'s leave them as they are originally, and instead \"roll\" the weights into a single square grid. The two representations below look different, but they are expressing the same equation, namely $$z=sum(wx)+b$$.

**[Figure: rolled weights]**

We\'ll now take a network trained to classify MNIST digits, except unlike in the last chapter, we will map directly from the input layer to the output layer with no hidden layer in between. Thus our network looks like this.

**[Figure: 1 layer MNIST]**

Let\'s visualize the learned weights feeding into the first output neuron, which is the one responsible for classifying the digit 0. The learned weights range from 0.1 to 0.9. We will color-code them so the lowest weight is black, and the highest is white.

**[Figure: rolled weights for 0-neuron]**

Squint your eyes a bit. Does it look a bit like a blurry 0? That\'s not a coincidence! The reason why this is so becomes clear if we think about what that neuron is doing. Because it is responsible for classifying 0s, it\'s \"goal\" (forgive the [anthropomorphization]()) is to output a high value for 0s and a lower one for non-0s. It can achieve this by having high values for weights corresponding to pixels which tend to be high in 0s, and lower weight values for pixels which tend to be low in 0s.

Let\'s look at the weights learned for all 10 of the output neurons. As suspected, they all appear to be somewhat blurry versions of our ten digits.

**[Figure: rolled weights for all 10 inputs, 2 rows]**

Even more satisfying is to watch them learned over iterations.
**[Figure: animated weights for all 10 inputs, 1 rows]**

So suppose we receive an input from the image of a 2. We can anticipate that the neuron responsible for classifying 2s will have a high value because its weights are such that high values tend to co-occur for pixels tending to be high in 2s--that is, they roughly line up. For other weight sets, _some_ of the weights will also line up with high-valued pixels, making their scores higher as well. However, there is much less overlap, and many of the high-valued pixels in those images will be nullified by low weights in the 2 neuron. The presence of the activation function does not change this, because it is monotonic with respect to the input, that is, the higher the input, the higher the output.

Thus, we can interpret these weights as forming \"templates\" of the output classes. This is really fascinating because we never _told_ our network anything in advance about what these digits are or what they mean, yet they resemble the object classes we are interested in anyway. This is a hint of what\'s really special about the internals of neural networks... they form _representations_ of objects, and it turns out these representations can be useful for much more than simple classification or prediction. We will take this representational capacity to the next level when we begin to study [convolutional neural networks]() but let\'s not get ahead of ourselves yet!

# 0op5, 1 d14 17 2ga1n

The last section raises many more questions than it provides answers, such as what happens to the weights when we add hidden layers? As we will see, this will build upon what we saw in the previous section in an intuitive way. But before we get to that, it will help to examine the performance of our neural network, and in particular, see what sorts of mistakes it tends to make.


Occasionally, our network will make a mistake that we can sympathize with. To my eye, it\'s not obvious that the digit to the bottom-left is a 6. I would have misclassified that as a 5 myself. But the bottom-right mistake is a more glaring error. Almost any person would immediately recognize that the ___ makes it a __, yet our machine misinterpreted it as a __. 


It probably did this 

# Breaking our neural network



p5 sketch + covnnet.js
from neural nets templates for mnist: we are learning a representation of a thing. a hint of cnn's ability to form representations which will be useful later


MNIST templates
CIFAR templates

What happens when we increase the number of layers?


https://cs231n.github.io/understanding-cnn/
http://cs.nyu.edu/~fergus/drafts/utexas2.pdf
http://arxiv.org/pdf/1312.6034v2.pdf
http://www.matthewzeiler.com/pubs/cvpr2010/cvpr2010.pdf
deepvis
mtyka recent stylenet
http://arxiv.org/pdf/1602.03616v1.pdf


2 layer softmax
now we see combination of higher level parts

deepvis, looks for text even though we didnt ask it

this is important because we are forming compositionality.




ideas
 - introduce CIFAR
 - demo of CIFAR like MNIST
 - MNIST confusion matrix