---
layout: chapter
title: "Generative models"
header_image: "/images/headers/DGN_places.jpg"
header_quote: "lovelace"
---


In the context of neural networks, generative models refers to those networks which output images. We've seen Deepdream and style transfer already, which can also be regarded as generative, but in contrast, those are produced by an optimization process in which convolutional neural networks are merely used as a sort of analytical tool. In generative models like autoencoders and generative adversarial networks, the convnets output the images themselves. This chapter will look at those two specifically.

## Autoencoders

So far, we've mostly interpreted neural networks as being predictive, i.e. given some inputs, what is the output of  -- where it's going, etc. But this is just a special case of a much more general capacity they have. 

https://medium.com/a-year-of-artificial-intelligence/lenny-2-autoencoders-and-word-embeddings-oh-my-576403b0113a#.k1ntx1xnr

Autoencoders
 - neural nets really more interesting in general capacity -- learning how to map desired x to y
 - encoder -> decoder
   - why???  world's most expensive identity function
 - "compression" of latent variable
 - denoising vs variational
 - variational = assume a probabilistic model interpretation (KL divergence)
   - useful because it makes the latent space move around
 - face examples (VRAE)
 
GANs

interesting property of DCGANs
 - generate tons of labeled images to fill out the image class manifold
 - then a nearest-neighbors classifier on that outperforms a RBF-SVM


hardmaru - GAN + VRAE making