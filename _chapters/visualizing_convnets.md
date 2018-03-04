---
layout: chapter
title: "Looking inside convnets"
---

<!--
http://www.cs.toronto.edu/~guerzhoy/321/lec/W07/HowConvNetsSee.pdf
https://www.youtube.com/watch?v=ghEmQSxT6tw
-->


{% include todo.html note="header=dog conv filters" %}


### Visualizing weights and activations

{% include figure_multi.md path1="/images/figures/alexnet-firstlayer-filters2.jpg" caption1="AlexNet" %}


### How to excite neurons

One simple way to get a hint of the features that neurons learn is by measuring their responses to test images. We feed many images to the network, and then for a particular neuron of interest, we extract patches of those images which maximally activated it. The following figures are taken from [_Visualizing and Understanding Convolutional Networks_](https://arxiv.org/abs/1311.2901) by Zeiler and Fergus.

{% include figure_multi.md path1="/images/figures/visualizing-convnets-zf_l1.png" caption1="Left: a set of nine 7x7 filters in the first convolutional layer of a trained network. Right: grid of 7x7 patches from actual images which maximally activated each of the nine filters." %}

What do we see in this experiment? It seems the resulting image patches resemble the filters. For example, for the top left filter, we receive image patches with strong diagonal slopes, and in the middle filter in the bottom row, we get dark green square patches. For at least the first layer, this should make intuitive sense given what we saw in the previous chapters; the more a patch is correlated to a filter, the stronger its activation in that filter will be, relative to other patches. 

But what about subsequent convolutional layers? The principle still applies, but we are currently interested in the original images, not the input volumes at those layers which get processed by the filters. By then, there has already been several layers of processing, so it doesn't make sense to compare the filters to the original images. Nevertheless, we can still perform the same experiment at later layers; we can input many images, measure the activations at later layers, and keep track of the image patches which generate the strongest response. Note that at later layers, the effective "receptive field" for a neuron is larger than the filter itself, due to pooling and convolution in previous layers. By the last layer of the network, the receptive field has grown to encompass the entire image.

{% include figure_multi.md path1="/images/figures/visualizing-convnets-zf_l2.png" caption1="Left: Same experiment but at layer 2 instead of layer 1. Each cell of 9 images shows the image patches which most activated a particular convolutional filter. Via [Zeiler and Fergus 2013](https://arxiv.org/abs/1311.2901)" %}

Above, we are looking at the experiment repeated at layer 2. The effective receptive field is now larger, and it no longer makes sense to compare them to the actual filters for each of those neurons. Strikingly though, we see that the image patches within each cell have much in common. For example, the cell at row 2, column 2, responds to several images that contain concentric rings, including what appears to be an eye. Similarly, row 2, column 4, has patches which contain many arcs. The last cell has hard perpendicular lines. It should be noted that these image patches often vary widely in terms of color distribution and lighting. Yet they are associated by sharing a particular feature which we can observe. This is the feature that neuron appears to be looking for.

{% include todo.html note="zeiler/fergus visualizing what neurons learn, image ROIs which maximally activate neurons" %}

### Occlusion experiments

{% include todo.html note="occlusion experiments, zeiler/fergus visualizing/understanding convnets https://cs231n.github.io/understanding-cnn/" %}



{% include figure_multi.md path1="/images/figures/activation_strength.png" caption1="AlexNet" %}


### Deconv / guided backprop

{% include todo.html note="deconv, guided backprop, deepvis toolbox" %}
{% include todo.html note="inceptionism class viz, deepdream" %}
deconvnets http://cs.nyu.edu/~fergus/drafts/utexas2.pdf
zeiler: https://www.youtube.com/watch?v=ghEmQSxT6tw
Saliency Maps and Guided Backpropagation on Lasagne https://github.com/Lasagne/Recipes/blob/master/examples/Saliency%20Maps%20and%20Guided%20Backpropagation.ipynb
multifaceted feature vis https://arxiv.org/pdf/1602.03616v1.pdf

{% include figure_multi.md path1="/images/figures/deepvis_freckles_small.jpg" caption1="AlexNet" %}

{::nomarkdown} 
<center><iframe width="720" height="405" src="https://www.youtube.com/embed/AgkfIQ4IGaM" frameborder="0" allowfullscreen></iframe></center>
{:/nomarkdown}


### Visualizing

aubun visualizing lenet classes http://www.auduno.com/2015/07/29/visualizing-googlenet-classes/
peeking inside convnets http://www.auduno.com/2016/06/18/peeking-inside-convnets/


### Neural nets are easily fooled


http://www.evolvingai.org/fooling + https://arxiv.org/pdf/1412.1897.pdf
https://arxiv.org/pdf/1412.6572v3.pdf
https://karpathy.github.io/2015/03/30/breaking-convnets/

{% include todo.html note="neural nets are easily fooled https://arxiv.org/abs/1412.1897" %}
{% include todo.html note="... but not on video" %}

### Etc

{% include todo.html note="notes on performance" %}
{% include todo.html note="attention, localization" %}

## Hint

Google cat -> deepdream

### Further reading

https://cs231n.github.io/understanding-cnn/

keras how convnets see the world https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html

http://yosinski.com/deepvis https://www.youtube.com/watch?v=AgkfIQ4IGaM

https://youtu.be/XTbLOjVF-y4?t=12m48s

https://jacobgil.github.io/deeplearning/class-activation-maps

http://arxiv.org/pdf/1312.6034v2.pdf

http://arxiv.org/pdf/1602.03616v1.pdf

{% include further_reading.md title="Slides: Deconvolutional networks and visualization" author="Matt Zeiler" link="http://cs.nyu.edu/~fergus/drafts/utexas2.pdf" %} 

{% include further_reading.md title="Understanding Neural Networks Through Deep Visualization" author="Jason Yosinski, Jeff Clune, Anh Nguyen, Thomas Fuchs, and Hod Lipson" link="http://yosinski.com/deepvis" %}
