---
layout: chapter
title: "Convolutional neural networks"
demo_includes: [mathjax, jquery, convnetjs, dataset, convnet, visualizer]
---

header: densecap

https://twitter.com/VisualGenome/status/682997407659982848


notes 
 - volumes should really be covered before this. this can review volumes
Convnets
 - intro, history
 - applications = links to other chapters
   - deepdream, styletransfer, go
 - weaknesses of neural networks
   - translation confusion
   - stretches, rotations, etc
 - convolution layer
   - convolution demo
   - convolution_all
 - pooling layers
 - review of volumes + the whole pipeline end to end
 - interpreting activations
 - hint of future chapters and applications


notes
 - have a section about feature representation. HOG/SIFT/handcrafted before + audio examples, replaced by learned features in deep learning


2010 (or 2009?) acero et al neural net for speech recognition swapped in for (gmm?) big boost on accuracy

2012 imagenet

alexnet = 15.4%
resnet '15 (152 layers) => ILSVRC 15 (he et al)
karen simonyan? vggnet
trends
 - removing fc + pooling layer for more convs
 - can remove pools by having convs w stride 2
 - smaller filters, more of them, and more layers

binary convnets

lenet
googlenet
alexnet
vggnet
resnet (microsoft)

batchnorm? 
localization - resnet 2015 dropped SOA from 25->9% localization error

---------

a common criticism of . one of the first concerted efforts to visualize convnets




Convolutional neural networks -- CNNs or convnets for short -- are at the heart of most applications of deep learning, having emerged in recent years as the most prominent strain of research within academia. They have revolutionized computer vision, achieving state-of-the-art benchmarks in a number of critical tasks, and have been widely deployed by tech companies for many of the new services and features we see today. They have numerous and diverse applications, including:

- recognizing and labeling objects, locations, and people in images
- converting speech to text and generating audio of natural sounds
- describing images and videos in natural language, as well as the reverse
- tracking roads and detecting obstacles in self-driving cars
- analyzing videogame screens to guide autonomous agents in them

Although convnets had been around since the 1980s and have their roots in earlier research into the visual cortex, they have only recently achieved fame for a series of spectacular results in important scientific problems across multiple domains. They extend neural networks by introducing two new kinds of layers, designed to improve the network's ability to cope with variations in position, scale, and viewpoint. Additionally, they will become increasingly deep, containing upwards of <a href="lecun">dozens</a> or even <a href="resnet">hundreds</a> of layers, forming detailed compositional models of images, sounds, as well as game boards and other spatial data structures.

Because of their success at vision-oriented tasks, they have been adopted by interactive and new media artists, allowing their installations not only to detect movement, but to actively identify, describe, and track objects in physical space.

The next few chapters will focus on convnets and their applications, with this one formulating them and how they ar trained, the [next one]() describing their properties, and subsequent chaptrs focusing on their creative and artistic applications.


## Why ordinary neural nets fail

To understand the innovations convnets offer, it helps to first review the weaknesses of ordinary neural networks, which are covered in more detail in the prior chapter [Looking inside neural networks](). 

Recall that in a trained one-layer ordinary neural network, the weights between the input pixels and the output neurons end up looking like templates for each output class. This is because they are constrained to capture all the information about each class in a single layer. Each of these templates looks like an average of samples belonging to that class.

{:.center}
![weights review](/images/figures/mnist_cifar_weights_review.png 'weights review')

In the case of the MNIST dataset, we see that the templates are relatively discernible and thus effective, but for CIFAR-10, they are much more difficult to recognize. The reason is that the image categories in CIFAR-10 have a great deal more internal variation than MNIST. Images of dogs may contain dogs which are curled up or outstretched, have different fur colors, be cluttered with other objects, and various other distortions. Forced to learn all of these variations in one layer, our network simply forms a weak template of dogs, and fails to accurately recognize unseen ones.

We can combat this by creating more hidden layers, giving our network the capacity to form a hierarchy of discovered features. For instance, we saw that many of the pictures of horses in CIFAR-10 are of left-facing and right-facing horses, making the above template resemble a two-headed horse. If we create a hidden layer, our network could potentially form learn a "right-facing horse" and "left-facing horse" template in the hidden layer, and the output neuron for horse could have strong weights to each of them. 

{% include todo.html note="multilayer weights" %} 

This makes some intuitive sense and gives our network more flexibility, but it's still problematic because of the extrme amount of permutations. In order to capture them, we'd need far too many neurons than we can practically afford to store or train. 

## Compositionality

Despite having their own fancy name, convnets are not categorically different from the neural networks we have seen so far. In fact, they inherit all of the functionality of those, and innovate those by introducing a new type of layer, namely a _convolutional layer_. Thus any neural network which contains at least one convolutional layer can be regarded as a convnet. Prior to this chapter, we've looked at seen _fully-connected layers_, in which each neuron is connected to every neuron in the previous layer. Convolutional layers break this assumption. We will formulate them shortly, but first a bit of intuition to motivate them.

Suppose I show you a picture of a car that you've never seen before. Chances are you'll be able to identify it as a car by observing that it is a permutation of the various properties of cars. In other words, the picture contains some combination of the parts that make up most cars, including a windshield, wheels, doors, and exhaust pipe. By recognizing each of the smaller parts and adding them up, you realize that this picture is of a car, despite having never encountered this precise combination of those parts. 

A convnet tries to do something similar: learn the individual parts of objects and store them in individual neurons, then add them up to recognize the larger object. This approach is advantageous for two reasons. One is that we can capture a greater variety of a particular object within a smaller number of neurons. For example, suppose we memorize 10 templates for different types of wheels, 10 templates for doors, and 10 for windshields. We thus capture $10 * 10 * 10 = 1000$ different cars for the price of only 30 templates. This is much more efficient than keeping around 1000 separate templates for cars, which contain much redundancy within them. But even better, we can reuse the smaller templates for different object classes. Wagons also have wheels. Houses also have doors. Ships also have windshields. We can construct a set of many more object classes as various combinations of these smaller parts, and do so very efficiently.


## Convolutional layers

The way we achieve this composition is using convolutional layers. Convolutional layers are actually mathematically very similar to fully-connected layers, differing only in the architecture. Let's first recall that in a fully-connected layer, we compute the value of a neuron $z$ as a weighted sum of all the previous layer's neurons, $z=b+\sum{w x}$.

{:.center}
![weights analogy](/images/figures/weights_analogy_2.png 'weights analogy')

We can interpret the set of weights as a _feature detector_ which is trying to detect the presence of a particular feature. We can visualize these feature detectors, as we did previously for MNIST and CIFAR. In a 1-layer fully-connected layer, the "features" are simply the the image classes themselves, and thus the weights appear as templates for the entire classes. 

In convolutional layers, we instead have a collection of smaller feature detectors--called _convolutional filters_-- which we individually slide along the entire image and perform the same weighted sum operation as before, on each subregion of the image. Essentially, for each of these small filters, we generate a map of responses--called an _activation map_--which indicate the presence of that feature across the image.

The process of convolving the image with a single filter is given by the following demo.

{% include todo.html note="rebuild mouse demo, button for changing filter/weight, click on filters" %}

{% include demo_insert.html width=960 height=540 path="/demos/demos/convolution.js" args="'MNIST',true" %}

In the above demo, we are showing a single convolutional layer on an MNIST digit. In this particular network at this layer, we have exactly 8 filters, and below we show each of the corresponding 8 activation maps.

{% include todo.html note="figure showing all the responses" %}

Each of the pixels of these activation maps can be thought of as a single neuron in the next layer of the network. Thus in our example, since we have 8 filters generating $25 x 25$ sized maps, we have $8 * 25 * 25 = 5000$ neurons in the next layer. The significance of each neuron is how present a small feature is in the image at a particular location.

Convolutional layers have a few properties, or hyperparameters, which must be set in advance. They include the size of the filters ($5x5$ in the above example), the stride. A full explanation of these is beyond the scope of the chapter.

## Pooling layers

Before we explain the significance of the convolutional layers, let's also introduce _pooling layers_, another kind of (much simpler) layer, which are very commonly found in convnets, often directly after convolutional layers.

pooling/subsampling layers obsolete now

{% include todo.html note="pooling layers image" %}

Pooling layers are beginning to gradually fall out of favor, being replaced.


## Volumes

Let's zoom out from what we just looked at and see the bigger picture. From this point onward, it helps to interpret 

{% include todo.html note="cs231n volume figure" %}


## Improving CIFAR-10 accuracy

{% include demo_insert.html width=960 height=540 path="/demos/demos/confusion_mnist.js" args="'CIFAR',true" %}


## Convnet architectures

{% include todo.html note="images of LeNet, ZFNet, ResNet" %}

{% include todo.html note="hyperparameter selection" %}

## Applications of convnets

{% include todo.html note="image localization, captioning, etc" %}

## Place in deep learning

{% include todo.html note="history, etc" %}







------------------------------------------

# etc

etc
lecun 89 -> first CNN according to zeiler (schmidhuber?)

CNNs offer us an an approach built on top of regular neural networks, which follows from the supposition made in the last paragraph. When they were first applied to handwritten digit classification, the system produced by Yann LeCun and co in 1991 improved handwritten digit classification to 99.3%, leading it to be used for reading around 10% of the checks deposited at banks in the U.S. throughout the 90s. But CNNs _really_ thrive when we expand our task to attempt to classify not just handwritten digits, but thousands of other kinds of objects appearing in pictures, from airplanes to lobsters to iPods and everything in between.


CNNs were the workhorses behind Deepdream, style transfer, and many other visual art applications of machine learning that appeared in 2015. Furthermore, also sound processing __

What innovations do CNNs have that makes them so effective at image classification? To understand them, let's first address the weaknesses of regular neural networks.

# Translation = confusion

When we first looked at the example of handwritten digit classification using regular neural networks, we made the following observation. Different images of the same numeral showed considerable variation in where the pixels lit up. Consider the following sample of 4s. The variations in them force our hidden layers to learn weights somewhat uncertainly, creating a filter that looks like an average of all of them. Such a filter helps to take into account all the variations in handwritten 4s, but is problematic between the filter weights of different numerals begin to overlap a great deal, creating confusion between them.

4 4 4 4 4 4 4 -> learned filter

Let's consider a handwritten 1 and 7. In practice, a 1 overlaps with the vertical bar in a 7 quite frequently. Because the weights in the 7-neuron are diluted, the top-bar receives a weaker signal. So in many cases, our system will confuse a 7 for a 1 or vice-versa. Similarly it will confuse 8 and 9, 5 and 6, and so on. Humans find distinguishing most people's 7s from their 1s easy - they simply observe that the top bar isn't there. But in regular neural networks, we don't have such a variable as \"top bars,\" we simply have the individual pixels. 


Perhaps, we would be able to build a better classifier if we _were_ able to represent objects encoded by multiple pixels, such as the \"top bars\" in 7s. We might characterize the 6 as a curved vertical arch and a loop at the bottom, and that a __ is a __.  Moreover, we could also say a cat is two eyes, a nose, some whiskers and ears, and so on. But let's not get ahead of ourselves yet.


notes
 - 3.08% top-5 on imagenet with residual connections + inception architecture http://arxiv.org/abs/1602.07261


convolution visual cortex responds to local receptive fields
convolution animation inspired from dl book show conv filter sliding across image generating dots

etc
 - http://karpathy.github.io/2015/10/25/selfie/
 - @fulhack typography


Starting in 2012, CNNs became the top-performing algorithms for the task of image classification, first doing so at the ILSVCR competition in which the SuperVision system designed by Krizhevsky and co achieved a 16% error on top-5 guesses, shattering the previous record by an unprecedented 10% margin. Since then, virtually all entries into the competition have been CNNs, and as of 2016, the current best sits at an astonishing 6%, [just 1% worse than a human](http://karpathy.github.io/2014/09/02/what-i-learned-from-competing-against-a-convnet-on-imagenet/)!

This dramatic improvement in our ability to design programs which accurately classify pictures has pushed the boundaries in a variety of higher-level applications, including:

---
cnns are the current top record holders in both image and speech classification. if you get through this you will fully understand at least in principle the current heavyweight champion algorithms known as of now in machine learning predictive  

cnns inherit everything from ordinary neural networks but build on top of them with several innovations which greatly improve its predictive accuracy as well as opening up a great many applications of cnns besides for ordinary regression and classification such as
---
