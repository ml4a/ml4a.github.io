---
layout: chapter
title: "Style transfer special cases"
---


In the [previous chapter](), we learned about the technique of [style transfer]() for recomposing images in the style of other images. Since the technique was first demonstrated by [Gatys et al](), there have been a number of notable follow-ups to the original research, which this chapter will survey. 

 * alternative techniques (style swap)
 * video style transfer
 * real-time style transfer
 * neural doodle
 * [assistive drawing tool](http://prostheticknowledge.tumblr.com/post/146031577846/stylit-assistive-creativity-research-from-dcgi-and)
 * style transfer in other domains? 
 * coloroless style transfer


## Restyling videos

A naive way to transfer style onto entire videos rather than just images is to simply process each frame individually. However, since the technique is not guaranteed to converge on the same image every time, features can appear and reappear in different parts of the original image, leading to an undesirable flicker artifact. 

To alleviate this, a technique was published by Ruder et al in April 2016 in the paper __. In it, they proposed a method using [optical flow](), a technique in computer vision to measures the displacement of objects in consecutive video frames. The method consists of pre-computing the optical flow among frames in the original video, then iteratively restyling all the frames simultaneously, with a third loss term based on trying to maintain the optical flow.


## Real-time style transfer

In April 2016, Johnson et al proposed a way to do style transfer which effectively sped it up by a factor of 1000, albeit reducing the quality. The way this was achieved was by training a second neural network to map from the original pixels to the restyled ones, thereby fitting the whole process into a single forward pass. 

This was quickly implemented by 


## In other domains?

To date, style transfer has been demonstrated only with images, but in a more general sense, it remains a topic of research in other domains.

https://www.audiolabs-erlangen.de/resources/MIR/2015-ISMIR-LetItBee

