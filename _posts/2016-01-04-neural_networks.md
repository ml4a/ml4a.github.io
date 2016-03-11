---
layout: post
title: "Neural networks"
date: 2016-01-04
---

[analytical engine]

\"It were much to be desired, that when mathematical processes pass through the human brain instead of through the medium of inanimate mechanism, it were equally a necessity of things that the reasonings connected with operations should hold the same just place as a clear and well-defined branch of the subject of analysis, a fundamental but yet independent ingredient in the science, which they must do in studying the engine.\" [ - Ada Lovelace](http://nlcatp.org/30-incredible-ada-lovelace-quotes/)

----

Nearly a century before neural networks were first conceived, Ada Lovelace described an ambition to build a \"calculus of the nervous system.\" Analogies between the brain and computers are as old as the philosophy of computation itself, and any __ inevitably led toward __. Ada would not live to see her dream come to fruition, as engineering at that time unable to replicate the complex circuitry the analytical engine required. Nevertheless, the idea was passed on to the next century when Alan Turing cited it as the inspiration in the Imitation Game, musing on the potential of algorithms to replicate human thinking. ..in the process of coining the field we now call artificial intelligence, kicking off .

Perceptrons could only fire 1s or 0s -- they were used to simulate and learn electrical circuits. [[needs more]]


## The once and future king

The resurgence of neural networks is a peculiar story. Intimately connected with the early days of AI, neural networks were formally derived ... since the 1950s when Frank Rosenblatt, building on the work of __, devised the Perceptron, a type of linear classifier that we saw in the last chapter. The Perceptron algorithm's ability to simulate and learn electrical circuits sparked the first boom of AI, inspiring a generation of science and science fiction alike. But the excitement was short lived, as these first neural networks over-promised their potential, leading to disappointment and a decline in academic and public interest. Until Marvin Minsky's 1961 book demonstrated various -- even trivial -- weaknesses in Perceptrons, it was assumed they would simply improve at the same pace as our computational power. Even Turing himself said machines would possess human-level intelligence by the year 2000 -- the year we had the Y2K scare. 

Despite a number of quiet but significant improvements to neural networks in the 80s and 90s, they remained on the sidelines through the 2000s, with most commercial and industrial applications of machine learning favoring support vector machines and various other approaches. Starting in 2009 and especially ramping up since 2012, neural networks have once again become the dominant breed of ML algorithms. Their resurgence was largely brought about by the emergence of convolutional and recurrent neural networks, which have surpassed (sometimes dramatically so) previous state-of-the-art methods in key problems in the audiovisual domain. But more interestingly, they have a number of new applications and properties not seen before, especially of a kind that has piqued the interest of artists and others from outside the AI field proper. This book will look more closely at convnets in particular several chapters from now.


## Scope

Neural networks are the only category of supervised learning algorithm this course will cover in detail for now. There are others, including support vector machines, decision trees, bayesian methods, and still more, but we will restrict our attention to neural networks because:

 - They have a simple and intuitive formulation which can be expressed using elementary math operations.
 - Deep neural networks currently represent the current state-of-the-art in several important machine learning tasks, the ones most relevant to this course.
 - Most of the recent creative uses of machine learning have been made using neural networks.


## From linear classifiers to neurons

Recall from the previous chapter that the input to a 2d linear classifier or regressor has the form

$$
\begin{eqnarray}
f(x_1, x_2) = b + w_1 * x_1 + w_2 * x_2
\end{eqnarray}
$$

More generally, in any number of dimensions, it can be expressed as

$$
\begin{eqnarray}
f(X) = b + \sum_i w_i * x_i
\end{eqnarray}
$$

In the case of regression, $$f(X)$$ gives us our predicted output, given the input vector $$X$$. In the case of classification, our predicted class is given by 

$$
\begin{eqnarray}
  \mbox{classification} & = & \left\{ \begin{array}{ll}
  	  1 & \mbox{if } f(X) \gt 0 \\
      0 & \mbox{if } f(X) \leq 0
      \end{array} \right.
\tag{1}\end{eqnarray}
$$

Each weight, $$w_i$$, can be interpreted as signifying the relative influence of the input that it\'s multiplied by, $$x_i$$. The threshold of the neuron is often called the _bias_, because it controls how predisposed the neuron is to firing a 1 or 0, irrespective of the weights. A high bias makes the neuron require a larger input to output a 1, and a lower one makes it easier. For this reason, we\ll refer to the threshold from now on as the bias and denote it as $$b$$.

We can get from this formula to a full-fledged neural network by introducing two innovations. The first is the addition of an _activation function_, which turns our linear discriminator into what\'s called a _neuron_, or sometimes a _unit_ (to dissociate with the brain analogy). The second is an architecture of neurons which are connected sequentially in _layers_. We will introduce these innovations in that order.

## Activation function

In both artificial and biological neural networks, a neuron does not just output the bare input it receives. Instead, there is one more step, called an _activation function_. The activation function takes the same weighted sum input from before, $$z = b + \sum_i w_i*x_i$$, and then transforms it once more before finally outputting it.

Many activation functions have been proposed, but for now we will describe two in detail: [sigmoid](_) and [ReLU](_). 

Historically, the sigmoid function is the oldest and most popular activation function. It is defined as:

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

Recall that $$e$$ denotes the [exponential constant](__), roughly equal to 2.718. A neuron which uses a sigmoid as its activation function is called a _sigmoid neuron_. We first set the variable $$z$$ to our original weighted sum input, and then pass that through the sigmoid function.

$$
z = b + \sum_i w_i * x_i \\
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

At first, this equation may seem complicated and arbitrary, but it actually has a very simple shape, which we can see if we plot the value of $$\sigma(z)$$ as a function of the input $$z$$. 

{:.center}
![sigmoid](/images/sigmoid.png 'sigmoid')


We can see that $$\sigma(z)$$ acts as a sort of \"squashing\" function, condensing our previously unbounded output to the range 0 to 1. In the center, where $$z = 0$$, $$\sigma(z) = 1/(1+exp^0) = 1/2$$. For large negative values of $$z$$, the term $$e^-z$$ and thus the denominator grow exponentially, and $$\sigma(z)$$ approaches 0. Conversely, large positive values of $$z$$ shrink $$exp^-z$$ to 0, so $$\sigma(z)$$ approaches 1.

e.g. sig(-5) = 0.006, sig(0) = 0.5, sig(5) = 0.993

Additionally, the sigmoid function was preferred because it is continuously differentiable, and its derivative, conveniently, is $$\sigma^\prime(z) = \sigma(z) * (1-\sigma(z))$$. This is important because we have to use calculus to train neural networks, but don\'t worry about that for now.

Sigmoid neurons were the basis of most neural networks for decades, but in recent years, they have fallen out of favor. The reason for this will be explained in more detail later, but in short, they have some unfavorable properties in networks which have many layers. Instead, most have shifted to using another type of activation function, the _rectified linear unit_, ReLU for short, introduced by Geoff Hinton et al. Despite its obtuse name, it is very simple.

$$
R(z) = max(0, z)
$$

In other words, it lets all positive values pass through unchanged, and sets any negative value to 0. We can see this if we graph it.

{:.center}
![ReLU](/images/relu.png 'ReLU')

Although newer activation functions are gaining traction in recent years, most deep neural networks use ReLU. Regardless of which activation function is used, we can represent a neuron in the form of a _compuational graph_ which shows the operations in order, giving us a nice intuitive visual representation of a neuron.

[ all the operations, + demo gif of all the numbers passing through ]

We can simplify this diagram like so, where the presence of the weights, bias, and activation function are encapsulated. This is the standard visualization of a neuron.

[ simplified neuron but with 3 inputs, simplified version of comp graph data flow ]

You may be wondering what the purpose of an activation function is, and why it is preferred to simply outputting the weighted sum, as is. The reason is that a weighted sum is _linear_ with respect to its inputs, i.e. it has a flat dependence on each of the inputs. In contrast, non-linear activation functions like these greatly expand the capacity of an ensemble of neurons to model curved or otherwise non-trivial functions. This will become clearer in the next section where we introduce the notion of _layers_.

# Neural networks are ____

Now that we have described neurons, we can now define neural networks. A neural network is composed of a series of _layers_ of neurons, such that the neurons in each layer connect to the neurons in the next layer. 

{:.center}
![neural network](/images/network.png 'neural network')

Note that when we count the number of layers in a neural network, we only count the layers with connections flowing into them (omitting our first, or _input layer_). Thus the above is a 2-layer neural network with 1 _hidden layer_. It contains 3 input neurons, 2 neurons in its hidden layer, and 1 output neuron.

Our computation starts with the input layer on the left, from which we pass the values to the hidden layer, and then in turn, the hidden layer will send its output values to the last layer, which contains our final value.

Note that it may look like the input neurons send out multiple values because each of them are connected to both of the neurons in the hidden layer. But really there is still only one output value per neuron, it just gets copied along each of its output connections. Neurons always output one value, no matter how many subsequent neurons it sends it to.

The process of a neural network sending an initial input forward through its layers to the output is called _forward propagation_ or a _forward pass_ and any neural network which works this way is called a _feedforward neural network_. As we shall soon see, there are some neural networks which allow data to flow in circles, but let\'s not get ahead of ourselves yet... 

Let\'s demonstrate a forward pass with this interactive demo. Click the \'Next\' button in the top-left corner to proceed. // You can see a forward pass in action in the following demo. Click 'manual' to pause the demo and go through the steps manually.

**[ DEMO: demo_forwardpass.html ]**


# More layers, more expressiveness

Why are hidden layers useful? The reason is that if we have no hidden layers and map directly from inputs to output, each input\'s contribution on the output is independent of the other inputs. In real-world problems, input variables tend to be highly interdependent and they affect the output in combinatorially complex ways. The hidden layer neurons allow us to capture interactions among our inputs which affect the final output downstream. 

Another way to interpret this is that the hidden layers represent higher-level \"features\" or attributes of our data. Each of the neurons in the hidden layer weigh the inputs differently, learning some different intermediary characteristic of the data, and our output neuron is then a function of these instead of the raw inputs. By including more than one hidden layer, we give the network an opportunity to learn multiple levels of abstraction of the original input data before arriving at a final output. This notion of high-level features will become more concrete in the next chapter when we look closely at the hidden layers.

Recall also that activation functions expand our capacity to capture non-linear relationships between inputs and outputs. By chaining multiple non-linear transformations together through layers, this dramatically increases the flexibility and expressiveness of neural networks. The proof of this is complex and beyond the scope of this book, but it can even be shown that any 2-layer neural network with a non-linear activation function (including sigmoid or ReLU) is a [_universal function approximator_](http://www.sciencedirect.com/science/article/pii/0893608089900208), that is it\'s theoretically capable of expressing any arbitrary input-to-output mapping. This property is what makes neural networks so powerful.

# Simple regression

Random set of dat, 3 cols. 1 regression value

Interactive 3 -> 1

Measure the error. Use L2 error


# Making neural networks predict MNIST

A simple way to structure a neural network so that it can classify an object is to have one output neuron for each of our classes or categories, input the data of a new sample, forward pass it through the network, observe all the class scores, and pick the one with the highest. Let\'s see this in action, using the example of handwritten digits which we have from the MNIST dataset.

About MNIST. 28x28, we can unroll the pixels into 784 inputs, one for ethe intensity of each pixel.

**[ FIGURE: fig_mnist.html ]**

The important thing to realize is that although this network seems a lot more imposing than our simple 3x2x1 network in the previous chapter, it works exactly as before, just with many more neurons. Each neuron in the first hidden layer receives


[interactive demo of mnist classifier with simple neural net]

final demo (these should be preloaded, asynchronously add 100 more if needed, make a stack)
 - MNIST image on left
 - connecting to pixels 784 input neurons
 - forward pass
 - 10 output neuron forward pass + softmax horizontal bar (green for correct, red for incorrect)

look for an incorrect one!


# Regression section (before MNIST)

We see that our small neural network, introduced above, maps a set of inputs to a single output during a forward pass. We can use this to make a prediction about a data point. 

[ interactive demo of us feeding dataset into a net and making a prediction ]

We have three examples of data.

Give random weights. Not very accurate.

Now a magic trick, I'll give it a new set of weights. Now let's run the examples again, and see that they are all correct now. 

the process of obtaining the correct weights is called training. for now, ignore it, it's a black box. we'll talk about it in the next section.

There are a number of important aspects about training -- you might have thought it's unfair that we predict training set -- after all it can just memorize them -- we'll get to this and other details of training in the [how they are trained].

# Classification

A simple way of turning this regressor and using it for binary classification is to use the sign. 

What if we have multiple classes? The way we would adapt our neural network is to have multiple output neurons, one for every class. Then, after a forward pass, our predicted classification is the class corresponding to the neuron which received the highest value at the end. This is called a \"softmax classifier.\"

[ image with say 3 classes, mouse/cat/dog, one prediction ]

# A real world example -- classifying handwritten digits

Let\'s take our small toy network from the previous section and adapt it to tackle a real application, that of classifying handwritten digits. (talk about significance of mnist )

The important thing to realize is that although this network seems a lot more imposing than our simple 3x2x1 network in the previous chapter, it works exactly as before, just with many more neurons. Each neuron in the first hidden layer receives

<> notes for this section
Some intuitions would be helpful at this point. Remember that in our toy linear classifier introduced in the previous chapter, we could interpret the score as being a weighted sum of our input variables, where the weights denote the influence of each input to the final score. So if a particular input variable has a positive correlation with the classifier, we should expect it would have a high positive weight. In this larger example, we now effectively have 10 classifiers (they are no longer linear, but the weight principle should remain true). 






# misc notes

neurons = outputs/units
multilayer perceptrons = misnomers

connection to neuro, synapses, neurons, axons. But most machine learning researchers and neuroscientists alike will warn you not to take this analogy to seriously! Though the design and terminology is inspired by the brain, neural networks will hardly give you much insight into how actual brains work. why?...

In research circles, sigmoid neurons have fallen out of favor in recent years because they they cause deep (many layer) neural networks to be prone to a problem called the \"vanishing gradient.\" We will address these later, but for now it is sufficient to proceed with sigmoid neurons because they are easy to understand, and we can swap them for other kinds of activation functions later without much fuss.

--
Originally, perceptrons were defined such that both the inputs and the output were binary numbers. They were used to model complex logic gate sequences, in spite of the comical fact that they could not model a simple XOR gate! In any case, the full expressive power of neural networks is found when we allow the inputs to take on continuous real-valued numbers, and from now on we will use them as such.

Let\'s apply it to [our simple classification problem introduced in the previous chapter].



notes
 - Tedd Hoff soon explored the option of just outputting the weight input in 1960 with “An adaptive “ADALINE” neuron using chemical “memistors” http://www.andreykurenkov.com/writing/a-brief-history-of-neural-nets-and-deep-learning/
 - the thinking machine youtube video: http://www.andreykurenkov.com/writing/a-brief-history-of-neural-nets-and-deep-learning/

