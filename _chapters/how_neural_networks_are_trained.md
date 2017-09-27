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


Imagine you are a mountain climber on top of a mountain, and night has fallen. You need to get to your base camp at the bottom of the mountain, but in the darkness with only your dinky flashlight, you can't see more than a few feet of the ground in front of you. So how do you get down? One strategy is to look in every direction to see which way the ground steeps downward the most, and then step forward in that direction. Repeat this process many times, and you will gradually go farther and farther downhill. You may sometimes get stuck in a small trough or valley, in which case you can follow your momentum for a bit longer to get out of it. Caveats aside, this strategy will eventually get you to the bottom of the mountain.

This scenario may seem disconnected from neural networks, but it turns out to be a good analogy for the way they are trained. So good in fact, that the primary technique for doing so, [_gradient descent_](https://en.wikipedia.org/wiki/Gradient_descent), sounds much like what we just described. 

Recall that _training_ refers to determining the best set of weights for maximizing a neural network's accuracy. In the previous chapters, we glossed over this process, preferring to keep it inside of a black box, and look at what already trained networks could do. In this chapter, we are going to look more closely at the process of training, and we shall see that it works much like the climber analogy we just described.

Neural networks can be used without knowing precisely how training works, just as one can operate a flashlight without knowing how the electronics inside it work. Most modern machine learning libraries have greatly automated the training process. Owing to those things and this topic being more mathematically rigorous, you may be tempted to set it aside and rush to applications of neural networks. But the intrepid reader knows this to be a mistake, not only because it gives valuable insights into how neural nets can be applied and reconfigured, but because the topic itself is one of the most interesting in research today. The ability to train large neural networks eluded us for many years and has only recently become feasible, making it one of the great success stories in the history of AI.

We'll get to gradient descent, backpropagation, and all the techniques involved in a few sections, but first, let's understand why training is hard to begin with. 

# Why training is hard

## A needle in a hyper-dimensional haystack

The weights of a neural network with hidden layers are highly interdependent. To see why, consider the highlighted connection in the first layer of the two layer network below. If we tweak the weight on that connection slightly, it will impact not only the neuron it propagates to directly, but also _all_ of the neurons in the next layer as well, and thus affect all the outputs. 

{% include todo.html note="figure with connection tweak" %}

For this reason, we know we can't obtain the best set of weights by optimizing one at a time; we will have to search the entire space of possible weight combinations simultaneously. How do we do this?

Let's start with the simplest, most naive approach to picking them: random guesses. We set all the weights in our network to random values, and evaluate its accuracy on our dataset. Repeat this many times, keeping track of the results, and then keep the set of weights that gave us the most accurate results. At first this may seem like a reasonable approach. After all, computers are fast; maybe we can get a decent solution by brute force. For a network with just a few dozen neurons, this would work fine. We can try millions of guesses quickly and should get a decent candidate from them. But in most real-world applications we have a lot more weights than that. Consider our handwriting example from [the previous chapter](ml4a/neural_networks/). There are around 12,000 weights in it. The best combination of weights among that many is now a needle in a haystack, except that haystack has 12,000 dimensions! 

You might be thinking that 12,000-dimensional haystack is "only 4,000 times bigger" than the more familiar 3-dimensional haystack, so it ought to take _only_ 4,000 times as much time to stumble upon the best weights. But in reality the proportion is incomprehensibly greater than that, and we'll see why in the next section. 

## n-dimensional space is a lonely place

If our strategy is brute force random search, we may ask how many guesses will we have to take before we obtain a reasonably good set of weights. Intuitively, we should expect that we need to take enough guesses to sample the whole space of possible guesses densely; with no prior knowledge, the correct weights could be hiding anywhere, so it makes sense to try to sample the space as much as possible.

To keep things simple, let's consider two very small 1-layer neural networks, the first one with 2 neurons, and the second one with 3 neurons. We are also ignoring the bias for the moment.

{% include todo.html note="2 and 3 neuron networks" %}

In the first network, there are 2 weights to find. How many guesses should we take to be confident that one of them will lead to a good fit? One way to approach this question is to imagine the 2-dimensional space of possible weight combinations and exhaustively search through every combination to some level of granularity. Perhaps we can take each axis and divide it into 10 segments. Then our guesses would be every combination of the two; 100 in all. Not so bad; sampling at such density covers most of the space pretty well. If we divide the axes into 100 segments instead of 10, then we have to make 100*100=10,000 guesses, and cover the space very densely. 10,000 guesses is still pretty small; any computer will get through that in less than a second. The following figure shows sampling two parameters to 10 and 100 bins.

{% include todo.html note="figure: sampling to 10 bins = 100 possible guesses and 100 bins = 1000 possible guesses" %}

How about the second network? Here we have three weights instead of two, and therefore a 3-dimensional space to search through. If we want to sample this space to the same level of granularity that we sampled our 2d network, we again divide each axis into 10 segments. Now we have 10 * 10 * 10 = 1,000 guesses to make. Both the 2d and 3d scenarios are depicted in the below figure. 

{% include todo.html note="roatate and label axes" %}
{% include figure_multi.md path1="/images/figures/sampling.png" caption1="Left: a 2d square sampled to 10% density requires 10² = 100 points. Right: a 3d cube sampled to 10% density requires 10³ = 1000 points." %}

1,000 guesses is a piece of cake, we might say. At a granularity of 100 segments, we would have $$100 * 100 * 100 = 1000000$$ guesses. 1,000,000 guesses is still no problem, but now perhaps we are getting nervous. What happens when we scale up this approach to more realistic sized networks? We can see that the number of possible guesses blows up exponentially with respect to the number of weights we have. In general, if we want to sample to a granularity of 10 segments per axis, then we need $$10^N$$ samples for an $$N$$-dimensional dataset. 

So what happens when we try to use this approach to train our network for classifying MNIST digits from the [first chapter](/ml4a/neural_networks/)? Recall that network has 784 input neurons, 15 neurons in 1 hidden layer, and 10 neurons in the output layer. Thus, there are $$784*15 + 15*10 = 11910$$ weights. Add 25 biases to the mix, and we have to simultaneously guess through 11,935 dimensions of parameters. That means we'd have to take $$10^{11935}$$ guesses... That's a 1 with almost 12,000 zeros after it! That is an unimaginably large number; to put it in perspective, there are only $$10^{80}$$ atoms in the entire universe. No supercomputer can ever hope to perform that many calculations. In fact, if we took all of the computers existing in the world today, and left them running until the Earth crashed into the sun, we still wouldn't even come close! And just consider that modern deep neural networks frequently have tens or hundreds of millions of weights.

This principle is closely related to what we call in machine learning [_the curse of dimensionality_](https://en.wikipedia.org/wiki/Curse_of_dimensionality). Every dimension we add into a search space exponentially blows up the number of samples we require to get good generalization for any model learned from it. The curse of dimensionality is more often applied to datasets; simply put, the more columns or variables a dataset is represented with, the exponentially more samples from that dataset we need to understand it. In our case, we are thinking about the weights rather than the inputs, but the principle remains the same; high-dimensional space is enormous!

{% include further_reading.md title="curse of dimensionality / eggshell example" %}

{% include todo.html note="clean up COD paragraph" %}

Obviously there needs to be some more elegant solution to this problem than random guesses, and indeed there are a number of them. Today, neural networks are generally trained using a variation of the gradient descent algorithm. To introduce the concept of gradient descent, we will again forget about neural networks for a minute, and start instead with a smaller problem, which we will scale up gradually.

# A simpler example first: linear regression

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

{% include todo.html note="change y to f(x) for clarity" %}

{% include figure_multi.md path1="/images/figures/lin_reg_randomtries.png" caption1="Three randomly-chosen line candidates" %}

Pretty clearly, the first two lines don't fit our data very well. The third one appears to fit a little better than the other two. But how can we decide this? Formally, we need some way of expressing how good the fit is, and we can do that by defining a _loss function_.

### Loss function

The loss function -- sometimes called a _cost function_ -- is a measure of the amount of error our linear regression makes on a dataset. Although many loss functions exist, all of them essentially penalize us on the distance between the predicted value of a given $$x$$ and its actual value in our dataset. For example, taking the line from the middle example above, $$ f(x) = -0.11 \cdot x + 2.5 $$, we highlight the error margins between the actual and predicted values with red dashed lines.

{% include figure_multi.md path1="/images/figures/lin_reg_error.png" caption1="" %}

One very common loss function is called _mean squared error_ (MSE). To calculate MSE, we simply take all the error bars, square their lengths, and take their average. 

$$ MSE = \frac{1}{n} \sum_i{(y_i - f(x_i))^2} $$

$$ MSE = \frac{1}{n} \sum_i{(y_i - (mx_i + b))^2} $$

We can go ahead and calculate the MSE for each of the three functions we proposed above. If we do so, we see that the first function achieves a MSE of 0.17, the second one is 0.08, and the third gets down to 0.02. Not surprisingly, the third function has the lowest MSE, confirming our guess that it was the line of best fit. 

We can get some intuition if we calculate the MSE for all $$m$$ and $$b$$ within some neighborhood and compare them. Consider the figure below, which uses two different visualizations of the mean squared error in the range where the slope $$m$$ is between -2 and 4, and the intercept $$b$$ is between -6 and 8.

{% include figure_multi.md path1="/images/figures/lin_reg_mse.png" caption1="Left: A graph plotting mean squared error for $ -2 \le m \le 4 $ and $ -6 \le p \le 8 $ <br/>Right: the same figure, but visualized as a 2-d <a href=\"https://en.wikipedia.org/wiki/Contour_line\">contour plot</a> where the contour lines are logarithmically distributed height cross-sections." %}

Looking at the two graphs above, we can see that our MSE is shaped like an elongated bowl, which appears to flatten out in an oval very roughly centered in the neighborhood around $$ (m,p) \approx (0.5, 1.0) $$. In fact, if we plot the MSE of a linear regression for any dataset, we will get a similar shape. Since we are trying to minimize the MSE, we can see that our goal is to figure out where the lowest point in the bowl lies.

### Adding more dimensions

The above example is quite minimal, having just one independent variable, $$x$$, and thus two parameters, $$m$$ and $$b$$. What happens when there are more variables? In general, if there are $$n$$ variables, a linear function of them can be written out as:

$$f(x) = b + w_1 \cdot x_1 + w_2 \cdot x_2 + ... + w_n \cdot x_n $$

Or in matrix notation, we can summarize it as:

$$
f(x) = b + W^T X
\;\;\;\;\;\;\;\;where\;\;\;\;\;\;\;\;
W = 
\begin{bmatrix}
w_1\\w_2\\\vdots\\w_n\\\end{bmatrix}
\;\;\;\;and\;\;\;\;
X = 
\begin{bmatrix}
x_1\\x_2\\\vdots\\x_n\\\end{bmatrix}
$$

One trick we can use to simplify this is to think of our bias $b$ as being simply another weight, which is always being multiplied by a "dummy" input value of 1. In other words, we let:

$$
f(x) = W^T X
\;\;\;\;\;\;\;\;where\;\;\;\;\;\;\;\;
W = 
\begin{bmatrix}
b\\w_1\\w_2\\\vdots\\w_n\\\end{bmatrix}
\;\;\;\;and\;\;\;\;
X = 
\begin{bmatrix}
1\\x_1\\x_2\\\vdots\\x_n\\\end{bmatrix}
$$

This is convenient both notationally, since now our function is more simply expressed as $f(x) = W^T X$, and conceptually, since we can now think of the bias as just another weight. In practice, it's just one more parameter that needs to be optimized, and it is not very different ________




Adding many more dimensions may seem at first to complicate our problem horribly, but it turns out that the formulation of the problem remains exactly the same in 2, 3, or any number of dimensions. Although it is impossible for us to draw it now, there exists a loss function which appears like a bowl in some number of dimensions -- a hyper-bowl! And as before, our goal is to find the lowest part of that bowl, objectively the smallest value that the loss function can have with respect to some parameter selection and dataset.

So how do we actually calculate where that point at the bottom is exactly? There are numerous ways to do so, with the most common approach being the [ordinary least squares](https://en.wikipedia.org/wiki/Ordinary_least_squares) method, which solves it analytically. When there are only one or two parameters to solve, this can be done by hand, and is commonly taught in an introductory course on statistics or linear algebra. 

Alas, ordinary least squares however cannot be used to optimize neural networks however, and so solving the above linear regression will be left as an exercise left to the reader. Instead we will introduce a much more powerful and general technique for solving both our linear regression, and neural networks: that of gradient descent.

### The curse of nonlinearity

Recall the essential difference between the linear equations we posed and a neural network is the presence of the activation function (e.g. sigmoid, tanh, ReLU, or others). Thus, whereas the linear equation above is simply $$y = b + W^T X$$, a 1-layer neural network with a sigmoid activation function would be $$f(x) = \sigma (b + W^T X) $$. 

This nonlinearity means that the parameters do not act independently of each other in influencing the shape of the loss function. Rather than having a bowl shape, the loss function of a neural network is more complicated. It is bumpy and full of hills and troughs. The property of being "bowl-shaped" is called [convexity](https://en.wikipedia.org/wiki/Convex_function), and it is a highly prized convenience in multi-parameter optimization. A convex loss function ensures we hav a global minimum (the bottom of the bowl), and that all roads downhill lead to it.

By introducing the nonlinearity, we give neural networks much more "flexibility" in modeling arbitrary functions, at the expense of losing this convenience. The price we pay is that there is no easy way to find the minimum in one step analytically anymore (i.e. by deriving neat equations for them). In this case, we are forced to use a multi-step numerical method to arrive at the solution instead. Although several alternative approaches exist, gradient descent remains the most popular and effective. The next section will go over how it works.

# Gradient Descent

The general problem we've been dealing with -- that of finding parameters to satisfy some objective function -- is not specific to machine learning. Indeed it is a very general problem found in [mathematical optimization](https://en.wikipedia.org/wiki/Mathematical_optimization), known to us for a long time, and encountered in far more scenarios than just neural networks. Today, many problems in multivariable function optimization -- including training neural networks -- generally rely on a very effective algorithm called _gradient descent_ to find a good solution much faster than taking random guesses. 


### The gradient descent method

Intuitively, the way gradient descent works is similar to the mountain climber analogy we gave in the beginning of the chapter. First, we start with a random guess at the parameters, and start there. We then figure out which direction the loss function steeps downward the most (with respect to changing the parameters), and step slightly in that direction. We repeat this process over and over until we are satisfied we have found the lowest point.

To figure out which direction the loss steeps downward the most, it is necessary to calculate the [_gradient_](https://en.wikipedia.org/wiki/Gradient) of the loss function with respect to all of the parameters. A gradient is a multidimensional generalization of a derivative; it is a vector containing each of the partial derivatives of the function with respect to each variable. In other words, it is a vector which contains the slope of the loss function along every axis. 

Although we've already said that the most convenient way to solve linear regression is via ordinary least squares or some other single-step method, let's quickly turn our attention back to linear regression to see a simple example of using gradient descent to solve a linear regression. 

Recall the mean squared error loss we introduced in the previous section, which we will denote as $J$.

$$ J = \frac{1}{n} \sum_i{(y_i - (mx_i + b))^2} $$

There are two parameters we are trying to optimize: $m$ and $b$. Let's calculate the partial derivative of $J$ with respect to each of them. 

$$ \frac{\partial J}{\partial m} = \frac{2}{n} \sum_i{x_i \cdot (y_i - (mx_i + b))} $$

$$ \frac{\partial J}{\partial b} = \frac{2}{n} \sum_i{(y_i - (mx_i + b))} $$

How far in that direction should we step? This turns out to be an important consideration, and in ordinary gradient descent, this is left as a hyperparameter to decide manually. This hyperparameter -- known as the _learning rate_ -- is generally the most important and sensitive hyperparameter to set and is often denoted as $$\alpha$$. If $$\alpha$$ is set too low, it may take an unacceptably long time to get to the bottom. If $$\alpha$$ is too high, we may overshoot the correct path or even climb upwards. 

Denoting the assignment operation as $:=$, we can write the update steps for the two parameters as follows.

$$ m := m - \alpha \cdot \frac{\partial J}{\partial m} $$

$$ b := b - \alpha \cdot \frac{\partial J}{\partial b} $$

If we take this approach to solving the simple linear regression we posed above, we will get something that looks like this:

{% include figure_multi.md path1="/images/figures/lin_reg_mse_gradientdescent.png" caption1="Example of gradient descent for linear regression with two parameters. We take a random guess at the parameters, and iteratively update our position by taking a small step against the direction of the gradent, until we are at the bottom of the loss function." %}

And if there are more dimensions? If we denote all of our parameters as $w_i$, thus giving us the form
$f(x) = b + W^T X $, then we can extrapolate the above example to the multimensional case. This can be written down more succinctly using gradient notation. Recall that the gradient of $J$, which we will denote as $\nabla J$, is the vector containing each of the partial derivatives. Thus we can represent the above update step more succinctly as:

$$ \nabla J(W) = \Biggl(\frac{\partial J}{\partial w_1}, \frac{\partial J}{\partial w_2}, \cdots, \frac{\partial J}{\partial w_N} \Biggr) $$

$$ W := W - \alpha \nabla J(W) $$

The above formula is the canonical formula for ordinary gradient descent. It is guaranteed to get you the best set of parameters for a linear regression, or indeed for any linear optimization problem. If you understand the significance of this formula, you understand "in a nutshell" how neural networks are trained. In practice however, certain things complicate this process in neural networks and the next section will get into how we deal with them.


# Applying gradient descent to neural nets

### The problem of convexity

In the previous section, we showed how to run gradient descent on a simple linear regression problem, and declared that doing so is guaranteed to find the correct parameters. This is true for optimizing a linear model as we did, but it's not true for neural networks, due to the nonlinearity introduced by their activation functions. Consequently, the loss function of a neural net is not "bowl-shaped", and it is not convex. Instead, its loss function is much more complex, with many hills and valleys and curves and other irregularities. This means there are many "local minima" i.e. parameterizations where the loss is the lowest in its immediate neighborhood, but not necessarily the absolute minimum (or "global minimum"). This means that if we run gradient descent, we might accidentally get stuck in a local minimum.

{% include figure_multi.md path1="/images/figures/non_convex_function.png" caption1="Example of non-convex loss surface with two parameters. Note that in deep neural networks, we're dealing with millions of parameters, but the basic principle stays the sam. Source: <a href=\"http://videolectures.net/site/normal_dl/tag=983679/deeplearning2015_bengio_theoretical_motivations_01.pdf\">Yoshua Bengio</a>." %}

For theoretical reasons beyond the scope of this book, it turns out that this is not a major problem in deep learning, because when there are enough hidden units alongside some other criteria, most local minima are "good enough," being reasonably close to the absolute minimum. According to [Dauphin et al](https://arxiv.org/abs/1406.2572), a bigger challenge than local minima are [saddle points](https://en.wikipedia.org/wiki/Saddle_point), along which the gradient becomes very close to 0. For an explanation of why this is true, see [this lecture](http://videolectures.net/deeplearning2015_bengio_theoretical_motivations/) by [Yoshua Bengio](http://www.iro.umontreal.ca/~bengioy/yoshua_en/) (beginning at section 28, 1:09:41).

Despite the fact that local minima are not a major problem, we'd still prefer to overcome them to the extent they are any problem at all. One way of doing this is to modify the way gradient descent works, which is what the next section is about.


### Stochastic, batch, and mini-batch gradient descent

Besides for local minima, "vanilla" gradient descent has another major problem: it's too slow. A neural net may have hundreds of millions of parameters; this means a single example from our dataset requires hundreds of millions of operations to evaluate. Subsequntly, gradient descent evaluated over all of the points in our dataset -- also known as "batch gradient descent" -- is a very expensive and slow operation. Moreover, because every dataset has inherent redundancy, it can be shown that a large enough subset of points can approximate the full gradient anyway, making batch gradient descent unnecessary.

It turns out that we can combat both this problem _and_ the problem of local minima using a modified version of gradient descent called [stochastic gradient descent (SGD)](https://en.wikipedia.org/wiki/Stochastic_gradient_descent). With SGD, we shuffle our dataset, and then go through each sample individually, calculating the gradient with respect to that single point, and performing a weight update for each. This may seem like a bad idea at first because a single example may be an outlier and not necessarily give a good approximation of the actual gradient. But it turns out that if do this for each sample of our dataset in some random order, the overall fluctations of the gradient update path will average out and converge towards a good solution. Moreover, SGD helps us get out of local minima and saddle points by making the updates more "jerky" and erratic, which can be enough to get unstuck if we find ourselves in the bottom of a valley. 

SGD is particularly useful in cases where the loss surface is especially irregular. But in general, the usual approach is to use what is called 
mini-batch gradient descent (MB-GD), in which the whole dataset is randomly subdivided into $$N$$ equally-sized mini-batches of $$K$$ samples each. $$K$$ may be a small positive number, or it can be in the dozens or hundreds; it depends on the specific architecture and application. Note that if $$K=1$$, then you have SGD, and if $$K$$ is the size of the whole dataset, it is batch gradient descent. 

With MB-GD, we have the best of both worlds; the gradient is smoother and more stable than SGD, and reasonably close to the full gradient, but we have a massive speed-up from not having to evaluate every sample in the dataset for each update. MB-GD is also computed very efficiently owing to parallelizable matrix operations. <!--Note: sometimes people use SGD to refer to both mini-batch gradient descent and one sample at a time.-->

{% include figure_multi.md path1="/images/figures/bumpy_gradient_descent.png" caption1="Example of gradient descent for non-convex loss function (such as a neural network), with two parameters $\theta_0$ and $\theta_1$. Source: <a href=\"http://www.holehouse.org/mlclass/01_02_Introduction_regression_analysis_and_gr.html\">Andrew Ng</a>." %}

In practice, MB-GD and SGD work well at efficiently optimizing the loss function of a neural network. However, they have weaknesses as well.

 - The aforementioned problem of saddle points; we can get stuck in a parameterization where the loss function plateaus, and the gradient gets very close to 0.
 - The learning rate remains a hyperparameter which must be set manually, which can be difficult to do. A learning rate which is too low leads to slow convergence, and one which is too high may overshoot the correct path. 

### Momentum

[Momentum](https://distill.pub/2017/momentum/) refers to a family of gradient descent variants where the weight update has inertia. In other words, the weight update is no longer a function of just the gradient at the current time step, but is gradually adjusted from the rate of the previous update. 

Recall that in standard gradient descent, we calculate the gradient $$\nabla J(W)$$ and use the following parameter update formula with learning rate $$\alpha$$. 

$$ W_{t} := W_{t} - \alpha \nabla J(W_{t}) $$

Note that we've appended the $$t$$ subscript to denote the current time step, which was previously omitted. In contrast, the generic formula for gradient descent with momentum is the following:

$$ z_{t} := \beta z_{t-1} + \nabla J(W_{t-1}) $$

$$ W_{t} := W_{t-1} - \alpha z_{t} $$

In the parameter update, we've replaced the gradient $$\nabla J(W_{t})$$ with a more complex function $$z_{t}$$ that takes into account the gradient in past time steps. The higher $$\beta$$ is set, the more momentum our parameter update is. If we set $$\beta = 0$$, then the formula reverts to ordinary gradient descent. $$\alpha$$ controls the overall learning rate of the process, as before.

You can think of the update path as being like a ball rolling downhill. Even if it gets to a region where the gradient changes significantly, it will continue going in roughly the same direction under its own momentum, only changing gradually along the path of the gradient. Momentum helps us escape saddle points and local minima by rolling out them from speed built up from previous updates. It also helps counteract against the common problem of zig-zagging found along locally irregular loss surfaces where the gradient steeps strongly along some directions and not others.

One alternative to the standard momentum formula is Nesterov accelerated gradient descent, given below:

$$ z_{t} := \beta z_{t-1} + \nabla J(W_{t-1} - \beta z_{t-1} ) $$

The only change is, rather than valuating the gradient where we currently are ($$W_{t-1}$$), we instead evaluate it at approximately where we will be at the next time step ($$W_{t-1} - \beta z_{t-1}$$), given the buildup of momentum carrying us in that direction. Calculating the gradient at that point instead of where we are currently lets us anticipate the loss surface ahead better and tune the momentum term accordingly. An illustration is given below:

{% include figure_multi.md path1="/images/figures/nesterov_acceleration.jpg" caption1="Nesterov momentum \"looks ahead\" to the approximate position we will be in the next update to calculate the gradient term in the update. Source: <a href=\"https://cs231n.github.io/neural-networks-3/\">Stanford CS231n</a>." %}

Momentum methods work pretty well, but like MB-GD and SGD use a single formula for the entire gradient, despite any internal asymmetries among parameters. In contrast, methods which adapt to each element in the gradient have some advantages, which will be looked at in the next section. The following article at [distill.pub](https://distill.pub) looks at momentum in much more mathematical depth and nicely illustrates why it works. 

{% include further_reading.md title="Why momentum works" author="Gabriel Goh" link="https://distill.pub/2017/momentum/" %} 

### Adaptive methods

Momentum comes in many flavors, and in general, finding fast, efficient, and accurate strategies for updating the parameters during gradient descent is a core objective of scientific research in the area, and a full discussion of them is out of the scope of this book. This section will instead quickly survey several of the more prominent variations in practical implementation, and refer to other materials online for a more comprehensive review.

One of the bigger annoyances in the training process is setting the learning rate $$\alpha$$. Typically, an initial $$\alpha$$ is set at the beginning, and is left to decay gradually over some number of time steps, letting it converge more precisely to a good solution. $$\alpha$$ is the same for each individual parameter.

This is unsatisfactory because it assumes that the learning rate must follow a set schedule which is identical for each individual parameter, irrespective of the particular characteristics of the loss surface at a given time step. Additionally, it's unclear how to set $$\alpha$$ and its decay rate in the first place. Momentum and Nesterov momentum help to reduce this burden by giving the update rate some dependence on local observations rather than the "one-size-fits-all" approach of vanilla gradient descent. Still, the choice of $$\alpha$$ and the inflexibility across parameters is seen as a problem.

A number of methods address this shortcoming by adapting the learning rate to each parameter individually, based on the assumption that there is a lot of variance of the loss across all the parameters. The simplest per-parameter update method is [AdaGrad](http://jmlr.org/papers/v12/duchi11a.html) (standing for "Adaptive subGradient"). With AdaGrad, each parameter is updated individually according to its own gradient, but with a new coefficient which attempts to equalize the learning rate between parameters which tend towards large gradients and those that tend to small ones. AdaGrad is defined in the following formula (Note: for the sake of avoiding confusion, note the subscript $$i$$ refers to index of the weight, rather than the time step as before
).

$$ w_{i} := w_{i} - \frac{\alpha}{\sqrt{G_{i}+\epsilon}} \frac{\partial J}{\partial w_{i}} $$

$$\sqrt{G_{i}+\epsilon}$$ represents the sum of the squares of the gradient for that paramter for each step since training began (the $$\epsilon$$ term is just some very small number, e.g. $$10^{-8}$$, to avoid division-by-zero). By dividing $$\alpha$$ for each parameter according to that quantity, we effectively slow down the learning rate for those parameters which have enjoyed large gradients up to that point, and conversely, speed up learning for parameters with minor or sparse gradients.

AdaGrad mostly eliminates the need to treat the initial learning rate $$\alpha$$ as a hyperparameter, but it has its own challenges as well. The typical problem with AdaGrad is that learning may stop prematurely as $$G_{i}$$ accumulates for each parameter over time and reduces the magnitude of the updates. A variant of AdaGrad, [AdaDelta](https://arxiv.org/abs/1212.5701), addresses this by effectivly restricting the window of the gradient accumulation term to the most recent updates. Another adaptive method which is very similar to AdaDelta is [RMSprop](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf). RMSprop -- proposed by [Geoffrey Hinton](http://www.cs.toronto.edu/~hinton/) during his Coursera class but otherwise unpublished -- similarly shortsights the update by summing the squares of the previous updates, but does so in a simpler way by using a standard [easing](http://easings.net) formula with a decay rate (which ends up being a hyperparameter). Thus, for both AdaDelta and RMSprop the update is not just adaptive with respect to parameters, but it's adaptive with respect to time as well, instead of having the learning rate decay monotonically until stopping.

### Adam and comparison of update methods

The last method worth mentioning in this chapter, and one of the most recent to be proposed, is [Adam](http://arxiv.org/abs/1412.6980), whose name is derived from adaptive moment estimation. Adam gives us the best of both worlds between adaptive methods and momentum-based methods. Like AdaDelta and RMSprop, Adam adapts the learning rate for each parameter according to a sliding window of past gradients, but it has a momentum component to smooth the path over time steps.

Still more methods exist, and a full discussion of them is out of the scope of this chapter. A more complete discussion of them, including derivations and practical tips, can be found in [this blog post by Sebastian Ruder](http://ruder.io/optimizing-gradient-descent/index.html).

This nice visualization, courtesy of [Alec Radford](https://twitter.com/alecrad), shows the characteristic behavior among the different gradient update methods discussed so far. Notice that momentum-based methods, Momentum and Nesterov accelerated gradient descent (NAG), tend to overshoot the optimal path by "rolling downhill" too fast, whereas standard SGD moves in the right path, but too slowly. Adaptive methods -- AdaGrad, AdaDelta, and RMSProp (and we could add Adam to it as well) -- tend to have the per-parameter flexibility to avoid both of those trappings.

{% include figure_multi.md path1="/images/figures/opt2a.gif" caption1="Contour plot of gradient update methods converging on good parameters. Figure by <a href=\"https://www.twitter.com/alecrad\">Alec Radford</a>" path2="/images/figures/opt1a.gif" caption2="Comparison of gradient update methods escaping from a saddle point. Notice that SGD gets stuck. Figure by <a href=\"https://www.twitter.com/alecrad\">Alec Radford</a>" %}

So which optimization method works best? There's no simple answer to this, and the answer largely depends on the characteristics of your data and other training constraints and considerations. Nevertheless, Adam has emerged as a promising method to at least start with. When data is sparse or unevenly distributed, the purely adaptive methods tend to work best. A full discussion of when to use each method is beyond the scop of this chapter, and is best found in the academic papers on optimizers, or in practical summaries such as [this one by Yoshua Bengio](https://arxiv.org/pdf/1206.5533v2.pdf).

For further reading on gradient descent optimization, see the following:

{% include further_reading.md title="An overview of gradient descent optimization algorithms" author="Sebastian Ruder" link="http://ruder.io/optimizing-gradient-descent/index.html" %} 

{% include further_reading.md title="Optimizing convolutional networks (CS231n)" author="Andrej Karpathy" link="https://cs231n.github.io/neural-networks-3/" %} 


# Hyperparameter selection and evaluation

### Cross-validation

### Parameter searching

grid param search
AutoML
NEAT

{% include further_reading.md title="link" author="?" link="https://cs231n.github.io/neural-networks-3/" %} 


## Overfitting and regularization

In all machine learning algorithms, including neural networks, there is a common problem which has to be dealt with, which is the problem of _overfitting_. 

Recall from the previous section that our goal is to minimize the error in unknown samples, i.e. the test set, which we do by setting the parameters in such a way that we minimize loss in our known samples (the training set). Sometimes we notice that we have low error in the training set, but the error in the test set is much higher. This suggests that we are _overfitting_, a phenomenon which is common to all machine learning algorithms and must be dealt with. Let's see an example.

The two graphs below show the same set of training samples observed, the blue circles. In both, we attempt to learn the best possible polynomial curve through them. The one on the left we see a smooth curve go through the points, accumulating some reasonable amount of error. The one on the right oscillates wildly but goes through all of the points precisely, accruing almost zero error. Ostensibly, the one on the right must be better because it has no error, but clearly something's wrong.

The one on left...

[1) smooth model] [2) wavy overfit model] (from bishop) 

The way we can think of overfitting is that our algorithm is sort of \"cheating.\" It is trying to convince you it has an artificially high score by orienting itself in such a way as to get minimal error on the known samples (since it happens to know their values ahead of time). 

It would be as though you are trying to learn how fashion works but all you've seen is pictures of people at disco nightclubs in the 70s, so you assume all fashion everywhere consists of nothing but bell bottoms, jean jackets, and __. Perhaps you even have a close family member whom this describes.

Researchers have devised various ways of combating overfitting (neural networks, not wardrobes). We are going to look at the few most important ones.




## Regularization

Regularization refers to imposing constraints on our neural network besides for just minimizing the error, which can generally be interpreted as \"smoothing\" or \"flattening\" the model function. As we saw in the polynomial fitting regression example, a model which has such wild swings is probably overfitting, and one way we can tell it has wild swings is if it has large coefficients (weights for neural nets). So we can modify our loss function to have an additional term to penalize large weights, and in practice, this is usually the following.

Use a penalty term
In the above example, we see we must have high coefficients. We want to penalize high coefficients. One way of doing that is by adding a regulariation term to the loss. One tha tworks well is L2 squared loss.  It looks like this.

We see that this term increases when the weights are large numbers, regardless if positive or negative. By adding this to our loss function, we give ourselves an incentive to find models with small w's, because they keep that term small.

But now we have a new dilemma. Mutual conflict between the terms.


### L1/L2 regularization

### Dropout

### Bathnorm

### preprocessing (norm, standard), weight init




# Backpropagation






------------






{% include further_reading.md title="Implementation of linear regression in python" author="Chris Smith" link="https://crsmithdev.com/blog/ml-linear-regression/" %} 


{% include further_reading.md title="link" author="?" link="https://betterexplained.com/articles/vector-calculus-understanding-the-gradient/" %} 

{% include further_reading.md title="link" author="?" link="http://ruder.io/optimizing-gradient-descent/" %} 


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

This is, in principle, what we have to do to solve a neural network. We have some loss function which expresses how poor or inaccurate our classifier is, and the loss is a function of the way we set our weights. In the neural network we drew above, there are 44 weights.

# Learning by data



# Loss function

Sum(L1 / L2 error)


Dropout

1) Training + test set

crucial. No supervised algo proceeds without it.
split into a test set. The reason why is that if we evaluate our ML algorithm's effectiveness on a set that it was also trained on, we are giving the machine an opportunity to just memorize the training set, basically cheating.  This won't generalize


2) Training + validation + test set

Dividing our data into a training set and test set may seem bulletproof, but it has a weakness: setting the hyperparameters. hyperparameters (personally I think they shoul gd have been called meta-parameters) are all the variables we have to set besides for the weights. Things like the number of hidden layers and how many neurons they have, the regularization strength, the learning rate, and others that are specific to various other algorithms.  

These have to be set before we begin training, but it's not obvious what the optimal numbers should be. So it may seem reasonable to try a bunch of them, train each of the resulting architectures on the same training set data, measure the error on the test set, and keep the hyperparameters which worked the best.

But this is dangerous because we risk setting the hyperparameters to be the values which optimize _that particular_ test set, rather than an arbitrary or unknown one.

We can get around this by partitioning our training data again -- now into a reduced training set and a _validation set_, which is basically a second test set where the labels are withheld. Thus we choose the hyperparameters which give us the lowest error on the validation set, but the error we report is still on the actual test set, whose true labels we have still never revealed to our algorithm during training time.



# 

So we use a training set and test set. 
But if we have hyperparameters (personally I think they should be called meta-parameters), we need to use a validation set as well. This gives us a second line of defense against overfitting.



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

https://distill.pub/2017/momentum/?utm_content=bufferd4ee6&utm_medium=social&utm_source=twitter.com&utm_campaign=buffer


https://crsmithdev.com/blog/ml-linear-regression/
http://briandolhansky.com/blog/artificial-neural-networks-linear-regression-part-1
https://spin.atomicobject.com/2014/06/24/gradient-descent-linear-regression/