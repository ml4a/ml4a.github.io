---
layout: chapter
title: "How neural networks are trained"
includes: [mathjax]
header_image: "/images/headers/topographic_map.jpg"
header_quote: "lovelace"
---
<!--
[http://www.cs.princeton.edu/courses/archive/spr08/cos598B/Readings/Fukushima1980.pdf]

http://www.summitpost.org/ruth-creek-topographic-map/771858

overview of grad descent algorithms: http://sebastianruder.com/optimizing-gradient-descent/index.html

http://www.holehouse.org/mlclass/01_02_Introduction_regression_analysis_and_gr.html
http://eli.thegreenplace.net/2016/understanding-gradient-descent/

http://ozzieliu.com/2016/02/09/gradient-descent-tutorial/
http://cs231n.github.io/optimization-2/

http://cs229.stanford.edu/notes/cs229-notes1.pdf
http://cs231n.github.io/neural-networks-3/

http://www.stat.ucdavis.edu/~chohsieh/teaching/ECS289G_Fall2015/lecture3.pdf

compare netwon's method / LBGFS

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


intro + mountain
why finding weights hard
 - hypercubes and sampling
 - curse of dimensionality + egg shell
gradient descent by backprop
 - simple example (bowl)
loss functions
 - MSE / quadratic cost (sum (y-a)^2)
cross-validation
regularization
 - L2
 - dropout

-->


Imagine you are a mountain climber on top of a mountain, and night has fallen. You need to get to your base camp at the bottom of the mountain, but in the darkness with your dinky flashlight, you can't see more than a few feet of the ground in front of you. So how do you get down? One strategy is to look in every direction to see which way the ground steeps downward the most, and then step forward in that direction. Repeat this process many times, and you will gradually go farther and farther downhill. You may sometimes get stuck in a small trough or valley, in which case you can back up a bit, or perhaps keep going in the same direction for a bit longer to get out of it, but in general, this strategy will slowly get you to the bottom of the mountain eventually.

This scenario may seem disconnected from neural networks, but it turns out to be a good analogy for the way they are trained. So good in fact, that one of the actual methods used, _gradient descent_, sounds much like what we just described. 

Recall that _training_ refers to determining the best set of weights for the network to get the most accuracy out of it. In the previous chapters, we glossed over this process, preferring to look at it as "magic" and look at what already trained networks could do. In this chapter, we are going to look more closely at the process of training, and we shall see that it works much like the climber analogy we just described.

Much of what's interesting about neural networks can be understood without knowing precisely how training works. Most modern machine learning libraries have greatly automated the training process. Owing to those things and this topic being more mathematically advanced (involving calculus), you may be tempted to skip this chapter, and indeed most of the remaining content in this book can be understood without it. But the intrepid reader is encouraged to proceed with this chapter, not only because it gives valuable insights into how to use neural nets, but because the topic itself is one of the most interesting in neural network research. The ability to train large neural networks eluded us for many years and has only recently become feasible, making it one of the great success stories in the history of AI.

We'll get to gradient descent in a few sections, but first, let's understand why choosing weights is hard to begin with. 

# Why training is hard

{% include todo.html note="hypercube/hypersphere" %}

## A needle in a hyper-dimensional haystack

The weights of a neural network with hidden layers are highly interdependent. To see why, consider the highlighted connection in the first layer of the two layer network below. If we tweak the weight on that connection slightly, it will impact not only the neuron it propagates to directly, but also _all_ of the neurons in the next layer as well, and thus affect all the outputs. 

{% include todo.html note="figure with connection tweak" %}

For this reason, we know we can't obtain the best set of weights one at a time; we will have to search the entire space of possible weight combinations simultaneously. How do we do this?

We'll start with the simplest, most naive approach to picking them: random guesses. We set all the weights in our network to random values, and evaluate its accuracy on some dataset. Repeat this many times, keeping track of the results, and then keep the set of weights that gave us the most accurate results. This seems to make some intuitive sense. After all, computers are fast; maybe we can get a decent solution by brute force. For a network with just a few dozen neurons, this would work fine. We can try millions of guesses quickly and should get a decent candidate from them. But in most real-world applications we have a lot more weights than that. Consider our handwriting example from [the previous chapter](ml4a/neural_networks/). There are around 12,000 weights in it. The best combination of weights among that many is now a needle in a haystack, except that haystack has 12,000 dimensions! 

You might be thinking that 12,000-dimensional haystack is "only 4,000 times bigger" than the more familiar 3-dimensional haystack, so it ought to take _only_ 4,000 times as much time to stumble upon the best weights. But in reality the proportion is incomprehensibly greater than that, and we'll see why in the next section. 

## n-dimensional space is a lonely place

If our strategy is brute force random search, we may ask how many guesses will we have to take to obtain a reasonably good set of weights. Intuitively, we should expect that we need to take enough guesses to sample the whole space of possible guesses densely; with no prior knowledge, the correct weights could be hiding anywhere, so it makes sense to try to sample the space as much as possible.

To keep things simple, let's consider two very small 1-layer neural networks, the first one with 2 neurons, and the second one with 3 neurons. We are also ignoring the bias for the moment.

{% include todo.html note="2 and 3 neuron networks" %}

In the first network, there are 2 weights to find. How many guesses should we take to be confident that one of them will lead to a good fit? One way to approach this question is to imagine the 2-dimensional space of possible weight combinations and exhaustively search through every combination to some level of granularity. Perhaps we can take each axis and divide it into 10 segments. Then our guesses would be every combination of the two; 100 in all. The figure below illustates this.

{% include todo.html note="figure: sampling to 10 bins, 100 possible guesses" %}

Not so bad. Sampling at such density covers most of the space pretty well. If we divide the axes into 100 segments instead of 10, then we have to make 100*100=10,000 guesses, and cover the space very densely. 10,000 guesses is still pretty small; any computer will get through that in less than a second. 

How about the second network? Here we have three weights instead of two, and therefore a 3-dimensional space to search through. If we want to sample this space to the same level of granularity that we sampled our 2d network, we again divide each axis into 10 segments. Now we have 10 * 10 * 10 = 1,000 guesses to make. This is depicted in the below figure. 

{% include todo.html note="figure: sampling to 10 bins, 1000 possible guesses" %}

1000 guesses is a piece of cake, we might say. At a granularity of 100 segments, we would have 100 * 100 * 100 = 1,000,000 guesses. 1,000,000 guesses is still no problem, but now perhaps we are getting nervous. What happens when we scale up this approach to more realistic sized networks? We can see that the number of possible guesses blows up exponentially with respect to the number of weights we have. In general, if we want to sample to a granularity of 10 segments per axis, then we need $$10^N$$ samples for an $$N$$-dimensional dataset. 

So what happens when we try to use this approach to train our network for classifying MNIST digits from the [first chapter](/ml4a/neural_networks/)? Recall that network has 784 input neurons, 15 neurons in 1 hidden layer, and 10 neurons in the output layer. Thus, there are 784*15 + 15*10 = 11,910 weights. Add 25 biases to the mix, and we have to simultaneously guess through 11,935 dimensions of parameters. That means we'd have to take $$10^11935$$ guesses... That's a 1 with almost 12,000 zeros after it! That is an unimaginably large number; to put it in perspective, there are only $$10^80$$ atoms in the entire universe. No supercomputer can ever hope to perform that many calculations. In fact, if we took all of the computers existing in the world today, and left them running until the Earth crashed into the sun, we still wouldn't even come close! And just consider that modern deep neural networks frequently have tens or hundreds of millions of weights.

##  The curse of dimensionality

This principle is closely related to what we call in machine learning [_the curse of dimensionality_](https://en.wikipedia.org/wiki/Curse_of_dimensionality). Every dimension we add into a search space exponentially blows up the number of samples we require to get good generalization for any model learned from it. The curse of dimensionality is more often applied to datasets; simply put, the more columns or variables a dataset is represented with, the exponentially more samples from that dataset we need to understand it. In our case, we are thinking about the weights rather than the inputs, but the principle remains the same; high-dimensional space is enormous!

{% include todo.html note="more on COD" %}

The gist of this exercise is that in order to represent the data well, we need to sample the space densely, or else we won't have enough information to model it accurately in its sparsest regions.

A vivid illustration of this is put forth by __. Sphere, egg, thin shell. !

In other words, 99.999% of the \"volume\" in hyper-sphere of 100 dimensions is enclosed in the tiny outer shell of it. Imagine if that were true of eggshells in eggs. 


Obviously there needs to be some more elegant solution to this problem than random guesses, and indeed there are a number of them. Today, neural networks are generally trained using backpropagation / grad descent.


<!--
Suppose you are a pollster and you are interviewing voters so you can better understand the political leanings of different demographics. At first, you are just collecting their age and income. To get a good cross section, you need to interview young and old, poor and rich alike, and each combination thereof. Add another axis, like education (e.g. how many years schooling), and your combinatorial space grows larger.

Let's divide each of these axes into 10 discrete bins. To get a representative dataset, we'd like to sample from every combination of bins. With just age and income (2d), that means we need to find 100 people. Adding the education axis, our requirement has ballooned up to 1000 samples.
-->

{% include todo.html note="roatate and label axes" %}
{% include figure.html path="/images/figures/sampling.png" caption="Left: a 2d square sampled to 10% density requires 10² = 100 points.<br/>Right: a 3d cube sampled to 10% density requires 10³ = 1000 points." %}


But the problem is scale; let's say we restrict our weights to be between 0 and 1 and sample them at intervals of 0.1, giving us 10 possible values each weight can take on. That means even in our basic network above, there are 10^86 possible weight combinations to try. That's a trillion times more than there are atoms in the universe! Now consider how many combinations there are to try in deep neural networks with hundreds of thousands or millions of weights...


# Gradient Descent

The problem we posed above of finding parameters to satisfy some objective function is not specific to machine learning. Indeed it is a very general problem found in [mathematical optimization](https://en.wikipedia.org/wiki/Mathematical_optimization), and the problem has been known to us for a long time. Today, many problems in multivariable function optimization -- including training neural networks -- generally rely on a very effective algorithm called _gradient descent_ to find a good solution much faster than taking random guesses. 

To introduce the concept of gradient descent, we will again forget about neural networks for a minute, and start instead with a smaller problem, which we will scale up gradually.

### A more simple example first: linear regression

Suppose we are given a set of 7 points, those in the chart to the bottom left. To the right of the chart is a scatterplot of our points.

{::nomarkdown}
<div style="text-align:center;">
	<div style="display:inline-block; vertical-align:middle; margin-right:100px;">
		<table width="200" style="border: 1px solid black;">
		  	<tbody>
				<tr>
					<td><script type="math/tex">x</script></td>
					<td><script type="math/tex">y</script></td>
				</tr>
				<tr><td>2.4</td><td>1.7</td></tr>
				<tr><td>2.8</td><td>1.85</td></tr>
				<tr><td>3.2</td><td>1.79</td></tr>
				<tr><td>3.6</td><td>1.95</td></tr>
				<tr><td>4.0</td><td>2.1</td></tr>
				<tr><td>4.2</td><td>2.0</td></tr>
				<tr><td>5.0</td><td>2.7</td></tr>
			</tbody>
		</table>
	</div>
	<div style="display:inline-block; vertical-align:middle;">
		<img src="/images/figures/lin_reg_scatter.png">
	</div>
</div>
{:/nomarkdown}

The goal of linear regression is to find a line which best fits these points. Recall that the general equation for a line is $$ f(x) = m \cdot x + b $$, where $$m$$ is the slope of the line, and $$b$$ is its y-intercept. Thus, solving a linear regression is determining the best values for $$m$$ and $$b$$, such that $$f(x)$$ gets as close to $$y$$ as possible. Let's try out a few random candidates.

{%include todo.html note="change y to f(x) for clarity" %}

{% include figure.html path="/images/figures/lin_reg_randomtries.png" caption="Three randomly-chosen line candidates" %}

Pretty clearly, the first two lines don't fit our data very well. The third one appears to fit a little better than the other two. But how can we decide this? Formally, we need some way of expressing how good the fit is, and we can do that by defining a _cost function_.

### Cost function

The cost is a measure of the amount of error our linear regression makes on a dataset. Although many cost functions have been proposed, all of them essentially penalize us on distance between the predicted value of a given $$x$$ and its actual value in our dataset. For example, taking the line from the middle example above, $$ f(x) = -0.11 \cdot x + 2.5 $$, we highlight the error margins between the actual and predicted values with red dashed lines.

{% include figure.html path="/images/figures/lin_reg_error.png" caption="" %}

One very common cost function is called _mean squared error_ (MSE), which is given by:

$$ MSE = \frac{1}{2n} \sum_i{(y_i - f(x_i))^2} $$

$$ MSE = \frac{1}{2n} \sum_i{(y_i - (mx_i + b))^2} $$

With MSE, we simply take all the error bars, square their lengths, and calculate their average. Actually, we take half the average (notice the $$ \frac{1}{2n} $$ multiplier), but this is simply done for convenience for when we have to take its derivative.

We can go ahead and calculate the MSE for each of the three functions we proposed above. If we do so, we see that the first function achieves a MSE of 0.17, the second one is 0.08, and the third gets down to 0.02. Not surprisingly, the third function has the lowest MSE. 

We can get some intuition if we calculate the MSE for all $$m$$ and $$b$$ within some neighborhood and compare them. Consider the figure below, which uses two different visualizations of the mean squared error in the range where the slope $$m$$ is between -2 and 4, and the intercept $$p$$ is between -6 and 8.

{%include todo.html note="change p to b, and multiply by 0.5" %}
{% include figure.html path="/images/figures/lin_reg_mse.png" caption="Left: A graph plotting mean squared error for $ -2 \le m \le 4 $ and $ -6 \le b \le 8 $ <br/>Right: the same figure, but visualized as a 2-d <a href=\"https://en.wikipedia.org/wiki/Contour_line\">contour plot</a> where the contour lines are logarithmically distributed height cross-sections." %}

Looking at the two graphs above, we can see that our MSE is shaped like an elongated bowl, which appears to flatten out in an oval very roughly centered in the neighborhood around $$ (m,b) \approx (0.5, 1.0) $$. In fact, if we plot the MSE of a linear regression for any dataset, we will get a similar shape. Since we are trying to minimize the MSE, we can see that our goal is to figure out where the lowest point in the bowl lies.

So how do we actually calculate that point at the bottom? If you have taken an introductory course in calculus, you know you can easily do it by solving for the [critical points](https://en.wikipedia.org/wiki/Critical_point_(mathematics)) where the partial derivatives of each parameter are 0 (so the function is therefore flat), and then simply keep whichever one has the lowest value. But in practice, when we have many more than two parameters, this turns out to be very computationally difficult to do. Things will get even more complicated when we fit points with neural networks because they are not linear and their cost functions are not simply bowl-shaped. Thus, the gradient descent algorithm was invented in order to arrive at the lowest point in a more reliable way across linear and nonlinear methods alike.

### How gradient descent proceeds

Intuitively, the way gradient descent works is similar to the mountain climber analogy we gave in the beginning of the chapter. First, we start with a random guess at the parameters. We then figure out which direction the cost function steeps downward the most, and step slightly in that direction. We repeat this process over and over until we are satisfied we have found the lowest point.

To figure out which direction the cost steeps downward the most, it is necessary to calculate the [_gradient_](https://en.wikipedia.org/wiki/Gradient) of the cost function with respect to all of the parameters. A gradient is a multidimensional generalization of a derivative; it is a vector containing each of the partial derivatives of the function with respect to each variable. In other words, it is a vector which contains the slope of the cost function along every axis. 

nice link: https://betterexplained.com/articles/vector-calculus-understanding-the-gradient/

calculate gradient

go down a bit, lather rinse repeat

{% include todo.html note="save locally" %}
![ andrew ng's image ](http://www.holehouse.org/mlclass/01_02_Introduction_regression_analysis_and_gr_files/Image%20[16].png)


## Complicating things a bit

### Neural networks are not linear

The linear regression we performed above gives 

In fact, there are methods for quickly computing the minimum analytically or numerically without doing gradient descent. But because of activation functions, neural nets are not linear, and their loss functions are not convex. 

[ bumpy image ]

### Local minima


### Stochastic gradient descent

Mini-batches




# Backpropagation

So now we know we can use gradient descent to solve for the weights of neural networks. Simply put, we calculate the gradient of the loss function with respect to the parameters, then do a small weight update in the direction of the gradient. But now we have another problem: how should we actually calculate the gradient? Naively, we can do it numerically using the 


 If we use Newton's method to numerically calculate the gradient, it would require us doing two forward passes for every single weight in our network to do a single weight update. If we have thousands or millions of weights, and need to do millions of weight updates to arrive at a good solution, there's no way this can take us a reasonable amount of time. Until we discovered the backpropagation algorithm and applied it successfully to neural networks, this was the main bottleneck preventing neural networks from achieving their potential.

So what is backpropagation? Backpropagation, or backprop for short, is short for "backward propagation of errors" 

, and it is the way we train neural networks, i.e. how we determine the weights. Although backprop does not guarantee finding the optimal solution, it is generally effective at converging to a good solution in a reasonable amount of time. 

The way backpropagation works is that you initialize the network with some set of weights, then you repeat the following sequence until you are satisfied with the network's performance: 

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

Let's reconnect what we have right now back to our analogy with the mountain climber. Suppose we have a simple network with just two weights. We can think of the the climber's position on the mountain, i.e. the latitude and longitude, as our two weights. And the elevation at that point is our network's loss with those two weight values. We can reduce the loss by a bit by adjusting our position slightly, in each of the two cardinal directions. Which way should we go?

Recall the following property of a line in 2d: [2d line, m * dx = dy]. In 3d, it is also true that m1 * dx + m2 * dy = dz. 

So let's say we want to reduce y by dy. If we calculate the slope m, we can find dx (use w instead?). One way to get this value is to calculate it by hand. But it turns out to be slow, and there is a better way to calculate it, analytically. The proof of this is elegant, but is outside the scope of this chapter. The following resources explain this well. Review [____] if you are interested.

other SGD explanations
1) Michael Nielsen
2) Karpathy's hackers NN as the computational graph
3) harder explanation (on youtube, i have the link somewhere...)
4) Simplest (videos which explain backprop)

//Once we have observed our loss, we calculate what's called the _gradient_ of our network. The gradient i


# AlecRad's gradient descent methods

different gradient descent methods
 - SGD
 - momentum
 - NAG
 - adagrad
 - adadelta
 - rmsprop


{% include figure.html path="/images/figures/opt2.gif" caption="Figure by <a href=\"https://www.twitter.com/alecrad\">Alec Radford</a>" %}



# Setting up a training procedure

Backpropagation, as we've described it, is the core of how neural nets are trained. From here, a few minor refinements are added to make a proper training procedure. The first is to separate our data into a training set and a test set. 

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

Dividing our data into a training set and test set may seem bulletproof, but it has a weakness: setting the hyper-parameters. Hyper-parameters (personally I think they shoul gd have been called meta-parameters) are all the variables we have to set besides for the weights. Things like the number of hidden layers and how many neurons they have, the regularization strength, the learning rate, and others that are specific to various other algorithms.  

These have to be set before we begin training, but it's not obvious what the optimal numbers should be. So it may seem reasonable to try a bunch of them, train each of the resulting architectures on the same training set data, measure the error on the test set, and keep the hyper-parameters which worked the best.

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

Chris Olah Backprop: http://colah.github.io/posts/2015-08-Backprop/
Chris Olah: neural net topology http://colah.github.io/posts/2014-03-NN-Manifolds-Topology/
Karpathy Neural nets for hackers http://karpathy.github.io/neuralnets/

backprop step by step example https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
step by step 2 http://experiments.mostafa.io/public/ffbpann/

https://www.inf.fu-berlin.de/inst/ag-ki/rojas_home/documents/tutorials/dimensionality.pdfs

http://cs231n.github.io/neural-networks-3/ 
alecrad 2 images

LBGFS, Adam

Gradient descent isn't the only way to solve neural networks. Notably, BGFS (or LBGFS when memory is limited) is sometimes used, but it operates on a similar principle: iterative, small weight updates convering on a good solution. 


implementation of gradient descent for linear regression: https://spin.atomicobject.com/2014/06/24/gradient-descent-linear-regression/


Image [16] : http://cs229.stanford.edu/notes/cs229-notes1.pdf

nice implementation: https://crsmithdev.com/blog/ml-linear-regression/