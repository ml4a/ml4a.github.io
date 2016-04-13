---
layout: post
title: "Style transfer"
date: 2016-02-04
---


# Style transfer


Similar to Deepdream, style transfer produces an image 

Style transfer is the technique of recomposing one image in the style of another, like the series of Mona Lisa regenerated in various iconic painting styles at the top of this page. The technique was first demonstrated in the paper "A neural algorithm of artistic style" published by Gatys et al at the University of Tubingen, and has since continued to be of great interest to both artists and scientists. Aside from autonomously generating stunning pictures (needs work), style transfer is a prized ability in AI across other domains as well. One could imagine a machine recomposing The Star Spangled Banner in heavy metal genre, or rewriting the novels of Elena Ferrante in the _frantic_ tone of Edgar Allen Poe. 

At a high level, style transfer, much like Deepdream, produces an image which is trying to optimize some function of a convolutional neural network's activations. More specifically, it is trying to simultaneously optimize the synthetic image's stylistic similarity with the input style image and its content similarity with the input content image, which we will describe in order.

The loss function for the content similarity is simply the distance between the synthetic image's activations and the style image's activations. In principle, this can be any or all of the activations, though in practice, most implementations choose one of the intermediate layers to compute this over.  

The loss function for style similarity is more tricky. Style is a nebulous term and is not easy to quantify. A number of approaches have been proposed, and a survey of them can be found in the 'Further Reading' section. This text will briefly describe the common approach taken by the first few implementations, ___. The style image is fed into the same convnet as the content image, after which we compute a Gram matrix for the activation volume in each layer of the convnet. A full explanation of a Gram matrix is out of the scope of this text, but for our purposes, it will suffice to say that this computation captures the covariance of each pair of activations in each volume.

# Style transfer in other domains




john mccarthy 1955 coined AI
turing 50 - computing machine and intelligence

graph of imagenet top accuracy, ILSVRC, Russakovsky et al 2015


convnet from the top -- big tiles to small tiles
single differentiable function


captioning: conditioning an RNN via convnet activations
from training in which we associate states of the convnet and rnn
learns more meaning iteratively

SHRDLU
 - put the box on a pyraimd

computer reads recipes and robot is trained to perform it
 - same crossmodal properties

olga russakovsky
richard socher