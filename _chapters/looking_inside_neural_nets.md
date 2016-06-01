---
layout: chapter
title: "Looking inside neural nets"
---

brainbow by katie matho


http://medicalxpress.com/news/2015-10-brain-cells.html img.medicalxpress.com/newman/gfx/news/hires/2015/1-researchersl.png

http://journal.frontiersin.org/article/10.3389/fnana.2014.00103/full

http://catalog.flatworldknowledge.com/bookhub/reader/127?e=stangor-ch03_s01 images.flatworldknowledge.com/stangor/stangor-fig03_003.jpg


In the [previous chapter](), we saw how a neural network can be trained to classify handwritten digits with a respectable accuracy of around 90%. In this chapter, we are going to evaluate its performance a little more carefully, as well as examine its internal state to develop a few intuitions about what's really going on. Later in this chapter, we are going to break our neural network altogether by attempting to train it on a more complicated dataset of objects like dogs, automobiles, and ships, to try to anticipate what kind of innovations will be necessary to take it to the next level.

 
# The template interpretation

[printblock analogy?] [borges quote about numbers]


Recall that when we input an image into our neural net, we "unroll" the pixels into a single row, as shown below. Rather than unrolling the pixels though, let's leave them as they are originally, and instead "roll" the weights into a single square grid. The two representations below look different, but they are expressing the same equation, namely $$z=b+\sum{w x}$$.

**[Figure: rolled weights + equivalent representation]**

x1 x2 x3     w1 w2 w3
x4 x5 x6  x  w4 w5 w6  = 
x7 x8 x9     w7 w8 w9

x1 w1
x2 w2
x3 w3
x4 w4
x5 w5
x6 w6
x7 w7
x8 w8
x9 w9

We'll now take a network trained to classify MNIST digits, except unlike in the last chapter, we will map directly from the input layer to the output layer with no hidden layer in between. Thus our network looks like this.

**[Figure: 1 layer MNIST]**

Let's visualize the learned weights feeding into the first output neuron, which is the one responsible for classifying the digit 0. We color-code them so the lowest weight is black, and the highest is white.

**[Figure: rolled weights for 0-neuron]**

Squint your eyes a bit. Does it look a bit like a blurry 0? That's not a coincidence! The reason why this is so becomes clear if we think about what that neuron is doing. Because it is responsible for classifying 0s, it's "goal" (forgive the [anthropomorphization]()) is to output a high value for 0s and a lower one for non-0s. It can achieve this by having high values for weights corresponding to pixels which _tend_ to be high in 0s, and lower weight values for pixels which tend to be low in 0s.

Let's look at the weights learned for all 10 of the output neurons. As suspected, they all appear to be somewhat blurry versions of our ten digits.

**[Figure: rolled weights for all 10 inputs, 2 rows]**

Even more satisfying is to watch them learned over iterations.

**[Figure: animated weights for all 10 inputs, 1 rows]**

So suppose we receive an input from the image of a 2. We can anticipate that the neuron responsible for classifying 2s will have a high value because its weights are such that high values tend to co-occur for pixels tending to be high in 2s--that is, they roughly line up. For other neurons, _some_ of the weights will also line up with high-valued pixels, making their scores somewhat higher as well. However, there is much less overlap, and many of the high-valued pixels in those images will be nullified by low weights in the 2 neuron. The presence of the activation function does not change this, because it is monotonic with respect to the input, that is, the higher the input, the higher the output.

We can interpret these weights as forming "templates" of the output classes. This is really fascinating because we never _told_ our network anything in advance about what these digits are or what they mean, yet they resemble the object classes we are interested in anyway. This is a hint of what's really special about the internals of neural networks... they form _representations_ of the objects they are trained on, and it turns out these representations can be useful for much more than simple classification or prediction. We will take this representational capacity to the next level when we begin to study [convolutional neural networks]() but let's not get ahead of ourselves yet!

This raises many more questions than it provides answers, such as what happens to the weights when we add hidden layers? As we will see, this will build upon what we saw in the previous section in an intuitive way. But before we get to that, it will help to examine the performance of our neural network, and in particular, see what sorts of mistakes it tends to make.

# 0op5, 1 d14 17 2ga1n

Occasionally, our network will make a mistake that we can sympathize with. To my eye, it's not obvious that the digit to the bottom-left is a 6. I would have misclassified that as a 5 myself. But the bottom-right mistake is a more glaring error. Almost any person would immediately recognize that the ___ makes it a __, yet our machine misinterpreted it as a __. 

**[FIGURE:: 2 mistakes]**

Let's look more closely at the performance of the last neural network of the previous chapter, which achieved 90% accuracy on MNIST digits. One way we can do this is by looking at a confusion matrix, a table which breaks down our predictions and performance. In the following confusion matrix, the 10 rows correspond to the actual labels of the MNIST dataset, and the columns represent the predicted labels. For example, the cell at the 3rd row and 5th column shows us that there were 27 instances in which an actual 3 was mislabeled by out neural network as a 5. The green diagonal of our confusion matrix shows us the quantities of correct predictions.

Hover your mouse over each cell to get a sampling of the top instances from each cell, ordered by the network's confidence (probability) for the prediction.

**[DEMO:: Confusion matrix with samples]**

We can also get a more summarial look by plotting the top sample for each cell of the confusion matrix, as seen below.

**[FIGURE:: Sample confusion matrix ]**

This gives us a nice idea of how the network learns to make certain kind of predictions. Looking at the last column, showing us predicted 0s, we see that it appears to be looking for big loops for 0s, and thin lines for 1s.

# Breaking our neural network

So far we've looked only at neural networks trained on handwritten digits. This gives us many insights but surely represents a best-case-scenario task. We have only ten classes, which are very well-defined and have relatively little internal variance. In most real-world applications, we are trying to classify images under much less ideal circumstances. Let's look at the performance of the same neural network on another dataset, [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html), a labeled set of 60,000 32x32 color images belonging to ten classes. The following is a random sample of images from CIFAR-10.

**[FIGURE:: Random sample from CIFAR-10]**

Right away, it's clear we must contend with the fact that these image classes differ in ways that digits don't. For example, cats can be facing different directions, have different colors and fur patterns, be outscretched or curled up, etc. Photos of cats will also be cluttered with other objects, further complicating the problem. 

Sure enough, if we train a 2-layer neural network on these images, our accuracy reaches only 28%. That's a great deal better than taking random guesses (which would get a 10% accuracy) but it's far short of the 90% our MNIST classifier achieves. When we start convolutional neural networks, we'll improve greatly on those numbers, for both MNIST and CIFAR-10. We can get a more precise sense about the shortcomings of ordinary neural networks by checking their weights.

Let's repeat the earlier experiment with observing the weights of a 1-layer neural network with no hidden layer, except on CIFAR-10. 

Compared to the MNIST weights, these have far less definition to them. The "templates" we saw with MNIST are far less evident (merge?). Certain details make intuitive sense, e.g. airplanes and ships are bluish on the outer edges of the images, reflecting the tendency for those images to have blue skies or waters around them. The weights image is after all, essentially an average of images belonging to that class, so we can expect blobby average colors to come out of the much less internally consistent image classes. But the "templates" we saw with MNIST are far less evident..

**[FIGURE:: CIFAR-10 1-layer weights ]**

Hidden layers are essential here. One obvious way they can help is best exemplified by the weight image for the horse class. The vague template of a horse is discernible, but it appears as though there is a head on each side of the horse. Evidently, the horses in CIFAR-10 seem to be usually facing one way or the other. If we create a hidden layer, a horse classifier could benefit by allowing the network to learn a "right-facing horse" or a "left-facing horse" inside the intermediate layer

**[FIGURE:: CIFAR-10 2-layer confusion matrix ]**


MNIST templates
CIFAR templates

# The composition interpretation

What happened to the weights when we added a hidden layer into our MNIST network? As in the demo of the last chapter, our MNIST net had a hidden layer of 15 neurons, which in turn connected to the final 10 output neurons. It's clear our simple template metaphor in the 1-layer network above doesn't work in this case, because we no longer have the 784 input pixels connecting directly to the output classes. 

In some sense, you might say we "forced" the first network to learn those templates because each of the weights associated with each pixel plugged into a single class label. In the more complicated network, the weights between the inputs and the hidden layer affect all of the output neurons. How should we expect the weights to look now?

Let's work backwards from our disadvvantage less info but higher level info -- we can do another set of forward passes on the dataset, and try to evaluate the behavior of the network by 

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

 Some intuitions would be helpful at this point. Remember that in our toy linear classifier introduced in the previous chapter, we could interpret the score as being a weighted sum of our input variables, where the weights denote the influence of each input to the final score. So if a particular input variable has a positive correlation with the classifier, we should expect it would have a high positive weight. In this larger example, we now effectively have 10 classifiers (they are no longer linear, but the weight principle should remain true). 
