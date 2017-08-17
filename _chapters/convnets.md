---
layout: chapter
title: "Convolutional neural networks"
includes: [mathjax, jquery, convnetjs, dataset, convnet, visualizer]
---

learn-features.png

umontreal image:
https://algobeans.com/2016/01/26/introduction-to-convolutional-neural-network/
-> http://www.iro.umontreal.ca/~bengioy/talks/DL-Tutorial-NIPS2015.pdf

http://blog.csdn.net/sparkkkk/article/details/65937088
<!--header: densecap


https://youtu.be/XTbLOjVF-y4?t=9m50s

https://twitter.com/VisualGenome/status/682997407659982848 -->

<!--
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


a common criticism of . one of the first concerted efforts to visualize convnets

-->


Convolutional neural networks -- CNNs or convnets for short -- are at the heart of deep learning, having emerged in recent years as the most prominent strain of research within academia. They have revolutionized computer vision, achieving state-of-the-art results in fundamental tasks, and have been widely deployed by tech companies for many of the new services and features we see today. They have numerous and diverse applications, including:

- recognizing and labeling objects, locations, and people in images
- converting speech into text and generating audio of natural sounds
- describing images and videos using natural language
- tracking roads and detecting obstacles in self-driving cars
- analyzing videogame screens to guide autonomous agents in them
- "hallucinating" generative images via image modeling

{% include todo.html note="pictures/citations for applications?" %}

Although convnets had been around since the 1980s and have their roots in earlier neuroscience research, they have only recently achieved fame in the wider scientific community for a series of remarkable successes in important scientific problems across multiple domains. They extend neural networks primarily by introducing a new kind of layer, designed to improve the network's ability to cope with variations in position, scale, and viewpoint. Additionally, they will become increasingly deep, containing upwards of dozens or even hundreds of layers, forming detailed compositional models of images, sounds, as well as game boards and other spatial data structures.

Because of their success at vision-oriented tasks, they have been adopted by interactive and new media artists, allowing their installations not only to detect movement, but to actively identify, describe, and track objects in physical spaces.

The next few chapters will focus on convnets and their applications, with this one formulating them and how they ar trained, the [next one]() describing their properties, and subsequent chaptrs focusing on their creative and artistic applications.


## Why ordinary neural nets fail

To understand the innovations convnets offer, it helps to first review the weaknesses of ordinary neural networks, which are covered in more detail in the prior chapter [Looking inside neural nets/](/ml4a/looking_inside_neural_networks/). 

Recall that in a trained one-layer ordinary neural network, the weights between the input pixels and the output neurons end up looking like templates for each output class. This is because they are constrained to capture all the information about each class in a single layer. Each of these templates looks like an average of samples belonging to that class.

{:.center}
![weights review](/images/figures/mnist_cifar_weights_review.png 'weights review')

In the case of the MNIST dataset, we see that the templates are relatively discernible and thus effective, but for CIFAR-10, they are much more difficult to recognize. The reason is that the image categories in CIFAR-10 have a great deal more internal variation than MNIST. Images of dogs may contain dogs which are curled up or outstretched, have different fur colors, be cluttered with other objects, and various other distortions. Forced to learn all of these variations in one layer, our network simply forms a weak template of dogs, and fails to accurately recognize unseen ones.

We can combat this by creating more hidden layers, giving our network the capacity to form a hierarchy of discovered features. For instance, we saw that many of the pictures of horses in CIFAR-10 are of left-facing and right-facing horses, making the above template resemble a two-headed horse. If we create a hidden layer, our network could potentially form learn a "right-facing horse" and "left-facing horse" template in the hidden layer, and the output neuron for horse could have strong weights to each of them. 

{% include todo.html note="multilayer weights" %} 

This makes some intuitive sense and gives our network more flexibility, but it's impractical for the network to be able to memorize the nearly endless set of permutations which would fully characterize a dataset of images. In order to capture this much information, we'd need far too many neurons for what we can practically afford to store or train. The advantage of convnets is that they will allow us to capture these permutations in a more efficient way.

## Compositionality

How can we encode variations among many classes of images efficiently? We can get some intuition to this question by considering an example.

Suppose I show you a picture of a car that you've never seen before. Chances are you'll be able to identify it as a car by observing that it is a permutation of the various properties of cars. In other words, the picture contains some combination of the parts that make up most cars, including a windshield, wheels, doors, and exhaust pipe. By recognizing each of the smaller parts and adding them up, you realize that this picture is of a car, despite having never encountered this precise combination of those parts. 

A convnet tries to do something similar: learn the individual parts of objects and store them in individual neurons, then add them up to recognize the larger object. This approach is advantageous for two reasons. One is that we can capture a greater variety of a particular object within a smaller number of neurons. For example, suppose we memorize 10 templates for different types of wheels, 10 templates for doors, and 10 for windshields. We thus capture $10 * 10 * 10 = 1000$ different cars for the price of only 30 templates. This is much more efficient than keeping around 1000 separate templates for cars, which contain much redundancy within them. But even better, we can reuse the smaller templates for different object classes. Wagons also have wheels. Houses also have doors. Ships also have windshields. We can construct a set of many more object classes as various combinations of these smaller parts, and do so very efficiently.

## Antecedents and inspirations to convnets

Before formally showing how convnets detect these kinds of features, let's take a look at some of the important precedents to them, to understand the evolution of our methods for combating the problems we described earlier.

### Experiments of Hubel & Wiesel (1960s)

During the 1960s, neurophysiologists [David Hubel](https://en.wikipedia.org/wiki/David_H._Hubel) and [Torsten Wiesel](https://en.wikipedia.org/wiki/Torsten_Wiesel) conducted a series of experiments to investigate the properties of the visual cortices of animals. In [one of the most notable experiments](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1359523/), they measured the electrical responses from a cat's brain while stimulating it with simple patterns on a television screen. What they found was that neurons in the early visual cortex are organized in a hierarchical fashion, where the first cells connected to the cat's retinas are responsible for detecting simple patterns like edges and bars, followed by later layers responding to more complex patterns by combining the earlier neuronal activities.

{:.center}
![Hubel + Wiesel](/images/figures/hubel-wiesel.jpg 'Hubel + Wiesel')

Later experiments on [macaque monkeys](http://www.cns.nyu.edu/~tony/vns/readings/hubel-wiesel-1977.pdf) revealed similar structures, and continued to refine an emering understanding of mammallian visual processing. Their experiments would provide an early inspiration to artificial intelligence researchers seeking to construct well-defined computational frameworks for computer vision.

[Hubel & Wisel: Receptive fields, binocular interaction and functional architecture in the cat's visual cortex](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1359523/)


### Fukushima's Neocognitron (1982)

Hubel and Wiesel's experiments were directly cited as inspiration by [Kunihiko Fukushima](http://personalpage.flsi.or.jp/fukushima/index-e.html) in devising the [Neocognitron](http://www.cs.princeton.edu/courses/archive/spr08/cos598B/Readings/Fukushima1980.pdf), a neural network  which attempted to mimic these hierarchical and compositional properties of the visual cortex. The neocognitron was the first neural network architecture to use hierarchical layers where each layer is responsible for detecting a pattern from the output of the previous one, using a sliding filter to locate it anywhere in the image.

{:.center}
![neocognitron](/images/figures/neocognitron.jpg 'neocognitron')

Although the neocognitron achieved some success in pattern recognition tasks, it was limited by the lack of a training algorithm to learn the filters. This meant that the pattern detectors were manually engineered for the specific task, using a variety of heuristics and techniques from computer vision. At the time, [backpropagation](/ml4a/how_neural_networks_are_trained/) had not yet been applied to train neural nets, and thus there was no easy way to optimize neocognitrons or reuse them on different vision tasks.

- [Fukushima's original paper](http://www.cs.princeton.edu/courses/archive/spr08/cos598B/Readings/Fukushima1980.pdf)
- [Scholarpedia article on neocognitron](http://www.scholarpedia.org/article/Neocognitron)


### LeNet (1998)

During the 1990s, a [team at AT&T Labs](https://www.youtube.com/watch?v=FwFduRA_L6Q) led by [Yann LeCun](http://yann.lecun.com/) trained a convolutional network, nicknamed ["LeNet"](http://yann.lecun.com/exdb/lenet/), to classify images of handwritten digits to an accuracy of 99.3%. Their system was used for a time to automatically read the numbers in 10-20% of checks printed in the US. LeNet had 7 layers, including 2 convolutional layers, with the architecture summarized in the below figure.

{:.center}
![neocognitron](/images/figures/lenet.png 'LeNet')

Their system was the first convolutional network to be applied to an industrial-scale application. Despite this triumph, many computer scientists believed that neural networks would be incapable of scaling to recognition tasks involving larger or more complex images. For this reason, machine learning would continue to be dominated by other algorithms for more than another decade.

### AlexNet (2012)

2010: ImageNet
2012: AlexNet
2013-onwards: 

{% include todo.html note="AlexNet section" %}

{% include figure_multi.md path1="/images/figures/alexnet.jpg" caption1="AlexNet" %}


## Convolutional layers

Despite having their own proper name, convnets are not categorically different from the neural networks we have seen so far. In fact, they inherit all of the functionality of those, and innovate upon it by introducing a new type of layer, namely a _convolutional layer_, emulating and refining the innovative structures of the neocognitron. Thus any neural network which contains at least one convolutional layer can be regarded as a convnet. Prior to this chapter, we've just looked at _fully-connected layers_, in which each neuron is connected to every neuron in the previous layer. Convolutional layers break this assumption.

Convolutional layers are actually mathematically very similar to fully-connected layers, differing only in the architecture. Let's first recall that in a fully-connected layer, we compute the value of a neuron $z$ as a weighted sum of all the previous layer's neurons, $z=b+\sum{w x}$.

{:.center}
![weights analogy](/images/figures/weights_analogy_2.png 'weights analogy')

We can interpret the set of weights as a _feature detector_ which is trying to detect the presence of a particular feature. We can visualize these feature detectors, as we did previously for MNIST and CIFAR. In a 1-layer fully-connected layer, the "features" are simply the the image classes themselves, and thus the weights appear as templates for the entire classes. 

In convolutional layers, we instead have a collection of smaller feature detectors--called _convolutional filters_-- which we individually slide along the entire image and perform the same weighted sum operation as before, on each subregion of the image. Essentially, for each of these small filters, we generate a map of responses--called an _activation map_--which indicate the presence of that feature across the image.

The process of convolving the image with a single filter is given by the following demo.

{% include todo.html note="rebuild mouse demo, button for changing filter/weight, click on filters" %}

{% include demo_insert.html path="/demos/convolution/" parent_div="post" %}

In the above demo, we are showing a single convolutional layer on an MNIST digit. In this particular network at this layer, we have exactly 8 filters, and below we show each of the corresponding 8 activation maps.

{% include demo_insert.html path="/demos/convolution_all/" parent_div="post" %}

Each of the pixels of these activation maps can be thought of as a single neuron in the next layer of the network. Thus in our example, since we have 8 filters generating $25 * 25$ sized maps, we have $8 * 25 * 25 = 5000$ neurons in the next layer. Each neuron signifies the amount of presence of a feature at a particular xy-point. It's worth emphasizing the differences in our visualization above to what we have seen before; in prior chapters, we always viewed the neurons (activations) of ordinary neural nets as one long column, whereas now we are viewing them as a set of activation maps. Although we could also unroll these if we wish, it helps to continue to visualize them this way because it gives us some visual understanding of what's going on. We will refine this point in a later section.

Convolutional layers have a few properties, or hyperparameters, which must be set in advance. They include the size of the filters ($5x5$ in the above example), the stride and spatial arrangement, and padding. A full explanation of these is beyond the scope of the chapter, but a good overview of these can be [found here](http://cs231n.github.io/convolutional-networks/).

## Pooling layers

Before we explain the significance of the convolutional layers, let's also quickly introduce _pooling layers_, another (much simpler) kind of layer, which are very commonly found in convnets, often directly after convolutional layers. These were originally called "subsampling" layers by LeCun et al, but are now generally referred to as pooling.

The pooling operation is used to downsample the activation maps, usually by a factor of 2 in both dimensions. The most common way of doing this is _max pooling_ which merges the pixels in adjacent 2x2 cells by taking the maximum value among them. The figure below shows an example of this.

{:.center}
![max pooling](/images/figures/max-pooling.png 'max pooling')

{% include todo.html note="citation/link" %}

The advantage of pooling is that it gives us a way to compactify the amount of data without losing too much information, and create some invariance to translational shift in the original image. The operation is also very cheap since there are no weights or parameters to learn.

Recently, pooling layers have begun to gradually fall out of favor. Some architectures have incorporated the downsampling operation into the convolutional layers themselves by using a stride of 2 instead of 1, making the convolutional filters skip over pixels, and result in activation maps half the size. These ["all-convolutional nets"](https://arxiv.org/abs/1412.6806) have some important advantages and are becoming increasingly common, but have not yet totally eliminated pooling.


## Volumes

Let's zoom out from what we just looked at and see the bigger picture. From this point onward, it helps to interpret the data flowing through a convnet as a volume. In previous chapters, our visualizations of neural networks always "unrolled" the pixels into a long column of neurons. But to visualize convnets properly, it makes more sense to continue to arrange the neurons in accordance to their actual layout in the image, as we saw in the last demo with the eight activation maps. 

In this sense, we can think of the original image as a volume of data. Let's consider the previous example. Our original image is 28 x 28 pixels and is grayscale (1 channel). Thus it is a volume whose dimensions are 28x28x1. In the first convolutional layer, we convolved it with 8 filters whose dimensions are 5x5x1. This gave us 8 activation maps of size 24x24. Thus the output from the convolutional layer is size 24x24x8. After max-pooling it, it's 12x12x8. 

What happens if the original image is color? In this case, our analogy scales very simply. Our convolutional filters would then also be color, and therefore have 3 channels. The convolution operation would work exactly as it did before, but simply have three times as many multiplications to make; the multiplications continue to line up by x and y as before, but also now by channel. So suppose we were using CIFAR-10 color images, whose size is 32x32x5, and we put it through a convolutional layer consisting of 20 filters of size 7x7x3. Then the output would be a volume of 26x26x20. The size in the x and y dimensions is 26 because there are 26x26 possible positions to slide a 7x7 filter into inside of a 32x32 image, and its depth is 20 because there are 20 filters.

{% include todo.html note="formula for xy size of volumes" %}

{:.center}
![volumes](/images/figures/cnn_volumes.jpg 'volumes')

We can think of the stacked activation maps as a sort-of "image."  It's no longer really an image of course because there are 20 channels instead of just 3. But it's worth seeing the equivalent representations; the input image is a volume of size 32x32x3, and the output from the first convolutional layer is a volume of size 26x26x20. Seeing the equivalence of these forms is crucial because it will help us understand the gist of the next section.

## Things get deep

Ok, here's where things are going to get really tricky! The whole chapter has been leading up to this section; we are going to design a full convolutional neural network to classify MNIST handwritten digits which will contain three convolutional layers and three pooling layers, followed by two fully connected layers. We are going to visualize the activations at each step of the way, and try to interpret what's going on.

{% include todo.html note="convnet w/ three convs visualization" %}

### What do multiple convolutional layers give us?

In the first convolutional layer, we deployed 8 activation feature detectors to find small multi-pixel patterns in the original image, giving us a volume of information corresponding to the presence of those features inside the image, which was subsequently pooled into a 12x12x8 resulting volume. Then we did another convolution on that volume. What is this second convolution achieving? Recall that the first conv is detecting patterns in the pixels of the original input image. In that case, it follows that the second conv is detecting patterns in the "pixels" of the volume resulting from the first conv (and pool). But those "pixels" aren't actually the original image pixels, but rather they signify the presence of the first layer features. So therefore, the second conv is detecting patterns among the features found


## Improving CIFAR-10 accuracy

{% include demo_insert.html path="/demos/confusion_cifar/" parent_div="post" %}


## Convnet architectures

{% include todo.html note="images of LeNet, ZFNet, ResNet" %}

{% include todo.html note="hyperparameter selection" %}

## Applications of convnets

{% include todo.html note="image localization, captioning, etc" %}

## Place in deep learning

{% include todo.html note="history, etc" %}






## etc

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


http://brohrer.github.io/how_convolutional_neural_networks_work.html




t-SNE own MD?
 - https://github.com/oreillymedia/t-SNE-tutorial
 - http://paperscape.org/
 - visualizing with https://indico.io/blog/visualizing-with-t-sne/

 keras transfer learning on images https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html


 semantic image classification https://github.com/torrvision/crfasrnn

 how to use t-SNE effectively: http://distill.pub/2016/misread-tsne/
 visualizing mnist with tsne http://colah.github.io/posts/2014-10-Visualizing-MNIST/

 colah convnets
 http://colah.github.io/posts/2014-07-Conv-Nets-Modular/
 http://colah.github.io/posts/2014-07-Understanding-Convolutions/
 http://colah.github.io/posts/2014-12-Groups-Convolution/

 localization
 http://cnnlocalization.csail.mit.edu/

 guide to convolution arithmetic https://arxiv.org/abs/1603.07285

tutorial on image segmentation with fcn http://warmspringwinds.github.io/tensorflow/tf-slim/2017/01/23/fully-convolutional-networks-(fcns)-for-image-segmentation/


## New writing

Before convolutional neural networks had taken over image classification, features were derived from hand-crafted edge-based feature detectors like SIFT and HOG. [Felzenszwalb, Girshick, Mcallester and Ramanan PAMI 2007 && Yan and Huang PASCAL 2010 classification](http://cs.nyu.edu/~fergus/drafts/utexas2.pdf)


# What else can convnets do?

detection - yolo