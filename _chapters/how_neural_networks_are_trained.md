---
layout: chapter
title: "How neural networks are trained"
header_image: "/images/headers/topographic_map.jpg"
header_quote: "lovelace"
---

[http://www.cs.princeton.edu/courses/archive/spr08/cos598B/Readings/Fukushima1980.pdf]

http://www.summitpost.org/ruth-creek-topographic-map/771858

overview of grad descent algorithms: http://sebastianruder.com/optimizing-gradient-descent/index.html

 - review
 - function graph + output
 - naive
   - try random ones
 - optimization, hill-climbing analogy
   - go smoothly
 - backprop
   - analytic way (chain rule)
   - [hackers way](karpathy.github.io/neuralnets/)
 - remember to
   - why sigmoid functions fell out of favor (vanishing grads)


-----------

http://cs231n.github.io/neural-networks-3/ 
alecrad 2 images

Mountains + backprop (analogy, picture, quotes)
[ mountains at night, or topographic map ]

Imagine you are a mountain climber on top of a mountain, and night has fallen. You need to get to your base camp at the bottom of the mountain, but in the darkness with your dinky flashlight, you can\'t see more than a few feet of the ground in front of you. So how do you get down? One strategy is to look in every direction to see which way the ground steeps downward the most, and then step forward slightly in that direction. Repeat this process many times, and you will gradually go further and further downhill. You may sometimes get stuck in a small trough or valley, in which case you can back up a bit, or perhaps keep going in the same direction for a bit longer to get out of it, but in general, this strategy will slowly get you to the bottom of the mountain eventually.

This scenario may seem disconnected from neural networks, but it turns out to be a good analogy for the way they are trained. So good in fact, that one of the actual methods used, _gradient descent_, sounds much like what we just described. 

Recall that _training_ refers to determining the best set of weights for the network to get the most accuracy out of it. In the previous chapters, we glossed over this process, preferring to look at it as \"magic\" and look at what already trained networks could do. In this chapter, we are going to look more closely at the process of training, and we shall see that it works much like the climber analogy we just described.

Much of what\'s interesting about neural networks can be understood without knowing precisely how training works. Most modern machine learning libraries have greatly automated the training process. Owing to those things and this topic being more mathematically advanced (involving calculus), you may be tempted to skip this chapter, and indeed most of the remaining content in this book can be understood without it. But the intrepid reader is encouraged to proceed with this chapter, not only because it gives valuable insights into how to use neural nets, but because the topic itself is one of the most interesting in neural network research. The ability to train large neural networks eluded us for many years and has only recently become feasible, making it one of the great success stories in the history of AI.

We'll get to gradient descent in a few sections, but first, let's understand why choosing weights is hard to begin with. 

[ hypercube ]

# A needle in a hyper-dimensional haystack

The weights of a neural network with hidden layers are highly interdependent. If we tweak a single weight, it will impact not only the neuron it propagates to directly, but also all the neurons in the next layer as well, and thus affect all the outputs. 

revision to above
--
Consider the connection highlighted in red seen in the following neural network with two hidden layers. So if we wish to make a small change to that weight to correct for some error found in one output neuron, we must consider how it will impact all the other neurons as well. We may wish to compensate for any unwanted changes by tweaking all those other weights as well. But for the same reason, doing so will change the network's behavior even more. Thus we see that weights are highly interdependent and changing them cannot be done in isolation.
--

[ highlight in red the pathways of a single weight change]. 

So we know we can\'t solve the weights one at a time; we have to search the entire space of possible weight combinations simultaneously. How do we do this?

We\'ll start with the simplest, most naive approach to picking them: random guesses. We set all the weights in our network to some random value, and evaluate its error. Repeat this many times, keeping track of the results, and then keep the set of weights that gave us the lowest error. After all, computers are fast; maybe we can get a decent solution by brute force. For a network with just a few dozen neurons, this would work fine. We can try millions of guesses quickly and should get a decent candidate from them. But in most real-world applications we have a lot more weights than that. Consider our handwriting example from [the previous chapter](__). There are around 12,000 weights in it. The best combination of weights among that many is now a needle in a haystack, except that haystack has 12,000 dimensions! 

You might be thinking that 12,000-dimensional haystack is only 4,000 times bigger than the more familiar 3-dimensional haystack, so it ought to take _only_ 4,000 times as much time to stumble upon the best weights. But in reality the proportion is immeasurably(?) greater than that, and we\'ll see why in the next section. 

# n-dimensional space is a lonely place

How many examples in a training set do we need to expect a reasonably good accuracy in a classifier or regressor? Let\'s use an example.


** replace pollster with just searching w-space. more connected.
** maybe curse of dimensionality moved into machine learning section

Suppose you are a pollster and you are interviewing voters so you can better understand the political leanings of different demographics. At first, you are just collecting their age and income. To get a good cross section, you need to interview young and old, poor and rich alike, and each combination thereof. Add another axis, like education (e.g. how many years schooling), and your combinatorial space grows larger.

Let\'s divide each of these axes into 10 discrete bins. To get a representative dataset, we\'d like to sample from every combination of bins. With just age and income, that means we need to find 100 people. Adding the education axis, our requirement has ballooned up to 1000 samples.

[ 2d bin sample] [ 3d bin sample ]

In general, if we want to sample to this level of precision, then we need $$10^N$$ samples for an $$N$$-dimensional dataset. In practice, we usually can\'t afford such precise requirements, but the gist of it is still true. In order to represent the data well, we need to sample the space densely, or else we won't have enough information to model it accurately in sparser regions.

In machine learning, we call this principle _the curse of dimensionality_. Simply put, adding more columns to your dataset blows up the number of data samples we require to get good generalization for any model learned from it.



A vivid illustration of this is put forth by __. Sphere, egg, thin shell. !

In other words, 99.999% of the \"volume\" in hyper-sphere of 100 dimensions is enclosed in the tiny outer shell of it. Imagine if that were true of eggshells in eggs. 


But the problem is scale; let's say we restrict our weights to be between 0 and 1 and sample them at intervals of 0.1, giving us 10 possible values each weight can take on. That means even in our basic network above, there are 10^86 possible weight combinations to try. That's a trillion times more than there are atoms in the universe! Now consider how many combinations there are to try in deep neural networks with hundreds of thousands or millions of weights...

the curse of dimensionality

Obviously there needs to be some more elegant solution to this problem. And the best one we've found so far is backpropagation.

# Backpropagation

Backpropagation stands for \"backward propagation of errors,\" or backprop for short, and it is the way we train neural networks, i.e. how we determine the weights. Although backprop does not guarantee finding the optimal solution, it is generally effective at converging to a good solution in a reasonable amount of time. 

The way backpropagation works is that you initialize the network with some set of weights, then you repeat the following sequence until you are satisfied with the network\'s performance: 

1) Take a batch of your data, run it through your network and calculate the error, that is the difference between what the network outputs and what we _want_ it to output, which is the correct values we have in our test set.

2) After the forward pass, we can calculate how to update the weights _slightly_ in order to reduce the error _slightly_. The update is determined by \"backward propagating\" the error through the network from the outputs back to the inputs.

3) Adjust the weights accordingly, then repeat this sequence.

Typically the loss of the network will look something like this with each round of this sequence. We typically stop this process once the network appears to be converging on some error.

The way backpropagation is actually implemented uses a method called _gradient descent_, which comes in a number of different flavors. We will look at the overarching method, and address the differences among them.

# Loss function

We have already seen the first step of this sequence -- forward propagating the data through the network and observing the outputs. Once we have done so, we quantify the overall error or \"loss\" of the network. This can be done in a few ways, with L2-loss being a very common loss function. 

L2-loss is {equations} 

If we change the weights very slightly, we should observe the L2-loss will change very slightly as well. And we want to change them in such a way that the loss decreases by as much as possible. So how do we achieve this?

# Descending the mountain

Let\'s reconnect what we have right now back to our analogy with the mountain climber. Suppose we have a simple network with just two weights. We can think of the the climber\'s position on the mountain, i.e. the latitude and longitude, as our two weights. And the elevation at that point is our network\'s loss with those two weight values. We can reduce the loss by a bit by adjusting our position slightly, in each of the two cardinal directions. Which way should we go?

Recall the following property of a line in 2d: [2d line, m * dx = dy]. In 3d, it is also true that m1 * dx + m2 * dy = dz. 

So let's say we want to reduce y by dy. If we calculate the slope m, we can find dx (use w instead?). One way to get this value is to calculate it by hand. But it turns out to be slow, and there is a better way to calculate it, analytically. The proof of this is elegant, but is outside the scope of this chapter. The following resources explain this well. Review [____] if you are interested.

other SGD explanations
1) Michael Nielsen
2) Karpathy's hackers NN as the computational graph
3) harder explanation (on youtube, i have the link somewhere...)
4) Simplest (videos which explain backprop)

//Once we have observed our loss, we calculate what\'s called the _gradient_ of our network. The gradient i


# AlecRad's gradient descent methods

different gradient descent methods
 - SGD
 - momentum
 - NAG
 - adagrad
 - adadelta
 - rmsprop



# Setting up a training procedure

Backpropagation, as we\'ve described it, is the core of how neural nets are trained. From here, a few minor refinements are added to make a proper training procedure. The first is to separate our data into a training set and a test set. 

Then cross validation GIF (taking combos of 5 -> train on 1)



# n-dimensional space is a lonely place (or t-SNE?)


validation set, cross-validation
regularization, overfitting
how to prevent overfitting
 - simple way is reduce representational power via fewer layers or neurons in hidden layers
 - better is using l2 or other kinds of regularization, dropout, input noise
 - neural nets w/ hidden layers are non-convex so they have local minima, but deep ones have less suboptimal (better) local minima than shallow ones


...


At this point, you may be thinking, "why not just take a big hop to the middle of that bowl?"  The reason is that we don't know where it is! Recall that 



Animation:

LHS varying slope of linear regressor, with vertical lines showing error
red bar showing amount of error
RHS graph error vs slope

2D analogue with jet-color 

In 3D, this becomes difficult to draw but the principle remains the same. We are going in the spatial direction towards our minimum point.

Beyond, we can't draw at all, but same principle.

So how does this apply to neural networks?

This is, in principle, what we have to do to solve a neural network. We have some cost function which expresses how poor or inaccurate our classifier is, and the cost is a function of the way we set our weights. In the neural network we drew above, there are 44 weights.

# Learning by data



# Cost function

Sum(L1 / L2 error)


# Overfitting

In all machine learning algorithms, including neural networks, there is a common problem which has to be dealt with, which is the problem of _overfitting_. 

Recall from the previous section that our goal is to minimize the error in unknown samples, i.e. the test set, which we do by setting the parameters in such a way that we minimize loss in our known samples (the training set). Sometimes we notice that we have low error in the training set, but the error in the test set is much higher. This suggests that we are _overfitting_, a phenomenon which is common to all machine learning algorithms and must be dealt with. Let's see an example.

The two graphs below show the same set of training samples observed, the blue circles. In both, we attempt to learn the best possible polynomial curve through them. The one on the left we see a smooth curve go through the points, accumulating some reasonable amount of error. The one on the right oscillates wildly but goes through all of the points precisely, accruing almost zero error. Ostensibly, the one on the right must be better because it has no error, but clearly something's wrong.

The one on left blah blah.

[1) smooth model] [2) wavy overfit model] (from bishop) 

The way we can think of overfitting is that our algorithm is sort of \"cheating.\" It is trying to convince you it has an artificially high score by orienting itself in such a way as to get minimal error on the known samples (since it happens to know their values ahead of time). 

It would be as though you are trying to learn how fashion works but all you've seen is pictures of people at disco nightclubs in the 70s, so you assume all fashion everywhere consists of nothing but bell bottoms, jean jackets, and __. Perhaps you even have a close family member whom this describes.

Researchers have devised various ways of combating overfitting (neural networks, not wardrobes). We are going to look at the few most important ones.

0) Regularization

Regularization refers to imposing constraints on our neural network besides for just minimizing the error, which can generally be interpreted as \"smoothing\" or \"flattening\" the model function. As we saw in the polynomial fitting regression example, a model which has such wild swings is probably overfitting, and one way we can tell it has wild swings is if it has large coefficients (weights for neural nets). So we can modify our loss function to have an additional term to penalize large weights, and in practice, this is usually the following.

Use a penalty term
In the above example, we see we must have high coefficients. We want to penalize high coefficients. One way of doing that is by adding a regulariation term to the loss. One tha tworks well is L2 squared loss.  It looks like this.

We see that this term increases when the weights are large numbers, regardless if positive or negative. By adding this to our loss function, we give ourselves an incentive to find models with small w's, because they keep that term small.

But now we have a new dilemma. Mutual conflict between the terms.

Dropout

1) Training + test set

crucial. No supervised algo proceeds without it.
split into a test set. The reason why is that if we evaluate our ML algorithm's effectiveness on a set that it was also trained on, we are giving the machine an opportunity to just memorize the training set, basically cheating.  This won't generalize


2) Training + validation + test set

Dividing our data into a training set and test set may seem bulletproof, but it has a weakness: setting the hyper-parameters. Hyper-parameters (personally I think they should have been called meta-parameters) are all the variables we have to set besides for the weights. Things like the number of hidden layers and how many neurons they have, the regularization strength, the learning rate, and others that are specific to various other algorithms.  

These have to be set before we begin training, but it\'s not obvious what the optimal numbers should be. So it may seem reasonable to try a bunch of them, train each of the resulting architectures on the same training set data, measure the error on the test set, and keep the hyper-parameters which worked the best.

But this is dangerous because we risk setting the hyper-parameters to be the values which optimize _that particular_ test set, rather than an arbitrary or unknown one.

We can get around this by partitioning our training data again -- now into a reduced training set and a _validation set_, which is basically a second test set where the labels are withheld. Thus we choose the hyper-parameters which give us the lowest error on the validation set, but the error we report is still on the actual test set, whose true labels we have still never revealed to our algorithm during training time.



# 

So we use a training set and test set. 
But if we have hyper-parameters (personally I think they should be called meta-parameters), we need to use a validation set as well. This gives us a second line of defense against overfitting.



misc content
===========
In the previous section, we introduced neural networks and showed an example of how they will make accurate predictions when they have the right combination of weights. But we glossed over something crucial: how those weights are actually found! The process of determining the weights is called _training_, and that is what the rest of this chapter is about.


This topic is more mathematically challenging than the previous chapters. In spite of that, our aim is to give a reader with less mathematical training an intuitive if not rigorous understanding of how neural networks are solved. If you are struggling with this chapter, know that it isn't wholly necessary for most of the rest of this book. It is sufficient to understand that there _is_ some way of training a network which can be treated like a black box. If you regard training as a black box but understand the architecture of neural networks presented in previous chapters, you should still be able to continue to the next sections.  That said, it would be very rewarding to understand how training works. May guide finer points, etc, plus it's mathematically elegant...  we will try to supplement the math with visually intuitive and analogies.

notes
because perceptron had step function, was non-differentiable, backprop came later

86 - backprop first applied to annns
rumelhart et al 86

hinton/salakhutinov 2006 - first deep network 
 - use unsupervised pre-training layer (restricted boltzman machine) by layer before using backprop
 - backprop doesn't work well from scratch (because of vanishing gradient?)



There are a number of important aspects about training -- you might have thought it's unfair that we predict training set -- after all it can just memorize them -- we'll get to this and other details of training in the [how they are trained].
