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

https://openai.com/blog/generative-models/

img: https://tryolabs.com/blog/2016/12/06/major-advancements-deep-learning-2016/
generative models: https://zo7.github.io/blog/2016/09/25/generating-faces.html
whats wrong with auto encoders https://danielwaterworth.github.io/posts/what's-wrong-with-autoencoders.html

http://web.mit.edu/vondrick/tinyvideo/

unreasonable confusion of VAEs https://jaan.io/unreasonable-confusion/

text to image https://github.com/paarthneekhara/text-to-image

GANS explained http://kvfrans.com/generative-adversial-networks-explained/

deep image completion http://bamos.github.io/2016/08/09/deep-completion/
seeing beyond edges of image http://liipetti.net/erratic/2016/12/11/seeing-beoynd-the-edges-of-the-image/

transfiguring portraits: http://homes.cs.washington.edu/~kemelmi/Transfiguring_Portraits_Kemelmacher_SIGGRAPH2016.pdf

soumith + yann https://code.facebook.com/posts/1587249151575490/a-path-to-unsupervised-learning-through-adversarial-networks/

end to end neural style with gans http://dmlc.ml/mxnet/2016/06/20/end-to-end-neural-style.html

eyescream eyescream

generating faces with torch http://torch.ch/blog/2015/11/13/gan.html

VAE https://ift6266h15.files.wordpress.com/2015/04/20_vae.pdf

CONDITIONAL IMAGE SYNTHESIS WITH AUXILIARY CLASSIFIER GANS https://arxiv.org/pdf/1610.09585.pdf
https://github.com/Evolving-AI-Lab/synthesizing

Fast Scene Understanding with Generative Models https://www.youtube.com/watch?v=4tc84kKdpY4 (nice video)

hardmaru images from latent vectors (GAN + VAE) blog.otoro.net/2016/04/01/generating-large-images-from-latent-vectors/

autoencoders book chapter http://ufldl.stanford.edu/tutorial/unsupervised/Autoencoders/

gen models -
describe probability distributions and data manifolds
2d line manifold, or plane in 3d, then go to eigenfaces
neural net modeling prob distribution
very sparse

we sometimes use the word astronomical to describe very large quantities. but no nimberassociated with astronmy, like the number of atoms in the universe, even begins to approach ___

GAN papers https://github.com/zhangqianhui/AdversarialNetsPapers

http://www.creativeai.net/posts/SFq7mQorZrw4cg6ij/artgan-artwork-synthesis-with-conditional-categorial-gans


https://medium.com/@devnag/generative-adversarial-networks-gans-in-50-lines-of-code-pytorch-e81b79659e3f#.okeu6cwzn

https://github.com/carpedm20/DiscoGAN-pytorch

https://tryolabs.com/blog/2016/12/06/major-advancements-deep-learning-2016/


http://gkalliatakis.com/blog/delving-deep-into-gans
https://github.com/ppwwyyxx/tensorpack/tree/master/examples/GAN