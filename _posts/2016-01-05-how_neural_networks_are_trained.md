---
layout: post
title: "How neural networks are trained"
date: 2016-01-04
---

picture of a mountain

 - review
   - function graph + output
     - example
   - 
   - turn back if you dont understand

 - naive
   - try random ones
 - optimization, hill-climbing analogy
   - go smoothly

 - backprop
   - analytic way
   - hackers way (karpathy)


 - remember to
   - why sigmoid functions fell out of favor (vanishing gradient)


different gradient descent methods
 - SGD
 - momentum
 - NAG
 - adagrad
 - adadelta
 - rmsprop


printable version with interactive parts broken out

going down a mountain but its dark. and you can only walk north south or east or west

http://karpathy.github.io/neuralnets/


LECTURE 5 how neural nets are trained history 6:00


backprop 86, but couldn't train well until 2006. so this is really important, and vindicated them


----
notes
because perceptron had step function, was non-differentiable, backprop came later

86 - backprop first applied to annns
rumelhart et al 86

hinton/salakhutinov 2006 - first deep network 
 - use unsupervised pre-training layer (restricted boltzman machine) by layer before using backprop
 - backprop doesn't work well from scratch (because of vanishing gradient?)


-----------

Learning by data

Cost function

-----------

Looking inside a neural network

In the previous chapter, 


-----------

Quote connecting mountains to backprop

START HERE

Imagine you are a mountain climber on top of a mountain, and night has fallen. You need to get to your base camp at the bottom of the mountain, but in the darkness with your dinky flashlight, you can\'t see more than a few feet of the ground in front of you. So how do you get down? One strategy is to look in every direction to see which way the ground steeps downward the most, and then step forward slightly in that direction. Repeat this process many times, and you will gradually go further and further downhill. You may sometimes get stuck in a small trough or valley, in which case you can back up a bit, or perhaps keep going in the same direction for a bit longer to get out of it, but in general, this strategy will slowly get you to the bottom of the mountain eventually.

This scenario may seem disconnected from neural networks, but it turns out to be a good analogy for the way they are trained. So good in fact, that the actual method used, _gradient descent_ sounds much like what we just described. Recall that _training_ refers to determining the best set of weights for the network to get the most accuracy out of it. In the previous chapters, we glossed over the training process, preferring to look at it as \"magic\" and look at what already trained networks could do. In this chapter, we are going to look more closely at the process of training.

Much of what\'s interesting about neural networks can be understood without knowing precisely how training works. Furthermore, most modern machine learning libraries have greatly automated the training process. Owing to those things and this topic being more mathematically advanced (involving calculus), you may be tempted to skip this chapter, and indeed most of the remaining content in this book can be understood without it. But the intrepid reader is encouraged to proceed with this chapter, not only because it gives valuable insights into how to use neural nets, but because the topic itself is highly interesting. The ability to train large neural networks eluded us for many years and has only recently become feasible, making it one of the great success stories of AI.

First, let\'s understand why choosing weights is hard to begin with. 

# A needle in a hyper-dimensional haystack

The weights of a neural network with hidden layers are highly inter-dependent. If we tweak a single weight, it will impact not only the neuron it propagates to directly, but also all the neurons in the next layer as well. [ graphic. caption (tweaking this weight) -> changes all of these neurons too]. So we know we can't solve the weights one at a time; we have to search the entire space of possible weight combinations. How do we do this?

We\'ll start with the simplest, most naive approach to picking them: random guesses. We can set all the weights in our network to some random value, forward propagate our dataset and measure the error. Repeat this many times, and then keep the set of weights that gave us the lowest error. After all, computers are fast; maybe we can just solve it by brute force. For a network with just a few dozen neurons, this would work fine. We can try millions of guesses quickly and should get a decent candidate from them. But in most real-world applications like our networks have a lot more than 6 neurons. Consider our handwriting example from before. There are around 12,000 weights. The best combination of weights among that many is now a needle in a haystack, except that haystack has 12,000 dimensions! You may be thinking that\'s only 4,000 times bigger than a normal haystack, but in reality, the proportion is  immeasurably greater than that, and is related to a concept called _the curse of dimensionality_. 

thin Shell in an egg

Consider the connection highlighted in red seen in the following neural network with two hidden layers. So if we wish to make a small change to that weight to correct for some error found in one output neuron, we must consider how it will impact all the other neurons as well. We may wish to compensate for any unwanted changes by tweaking all those other weights as well. But for the same reason, doing so will change the network's behavior even more. Thus we see that weights are highly interdependent and changing them cannot be done in isolation.

So how then do we find the best set of weights? One way we can think to do it would be to simply set all the weights to a random value and evaluate its accuracy. We can do this over and over and keep track of which set of weights gives us the lowest accuracy. We know computers are fast; why not just try to get a good solution using brute force? But the problem is scale; let's say we restrict our weights to be between 0 and 1 and sample them at intervals of 0.1, giving us 10 possible values each weight can take on. That means even in our basic network above, there are 10^86 possible weight combinations to try. That's a trillion times more than there are atoms in the universe! Now consider how many combinations there are to try in deep neural networks with hundreds of thousands or millions of weights...

the curse of dimensionality

Obviously there needs to be some more elegant solution to this problem. And the best one we've found so far is backpropagation.

# Backpropagation

Backpropagation stands for \"backward propagation of errors,\" or backprop for short. Although backprop does not guarantee finding the optimal solution, it is generally effective at converging to a good solution in a reasonable amount of time.

To help us understand how backprop works, let\'s build up to it by first considering a separate problem.


 - stochastic gradient descent (mini-batches)
 - validation, test set


# n-dimensional space is a lonely place (or t-SNE?)

...


At this point, you may be thinking, "why not just take a big hop to the middle of that bowl?"  The reason is that we don't know where it is! Recall that 

Let's make an analogy to see this more concretely. Suppose you are a mountaineer trying to get down to the base of a large mountain range. But because you are surrounded by various smaller peaks and valleys and plateaus, you may not see the base from your vantage point. All you see is the immediate neighborhood around you. So we take a small step down the hill we are on.



Animation:

LHS varying slope of linear regressor, with vertical lines showing error
red bar showing amount of error
RHS graph error vs slope

2D analogue with jet-color 

In 3D, this becomes difficult to draw but the principle remains the same. We are going in the spatial direction towards our minimum point.

Beyond, we can't draw at all, but same principle.

So how does this apply to neural networks?

This is, in principle, what we have to do to solve a neural network. We have some cost function which expresses how poor or inaccurate our classifier is, and the cost is a function of the way we set our weights. In the neural network we drew above, there are 44 weights.


# Cost function

Sum(L1 / L2 error)





misc content
===========
In the previous section, we introduced neural networks and showed an example of how they will make accurate predictions when they have the right combination of weights. But we glossed over something crucial: how those weights are actually found! The process of determining the weights is called _training_, and that is what the rest of this chapter is about.


This topic is more mathematically challenging than the previous chapters. In spite of that, our aim is to give a reader with less mathematical training an intuitive if not rigorous understanding of how neural networks are solved. If you are struggling with this chapter, know that it isn't wholly necessary for most of the rest of this book. It is sufficient to understand that there _is_ some way of training a network which can be treated like a black box. If you regard training as a black box but understand the architecture of neural networks presented in previous chapters, you should still be able to continue to the next sections.  That said, it would be very rewarding to understand how training works. May guide finer points, etc, plus it's mathematically elegant...  we will try to supplement the math with visually intuitive and analogies.