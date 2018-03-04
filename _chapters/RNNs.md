---
layout: chapter
title: "Recurrent neural networks"
header_image: "/images/headers/ascending_descending.jpg"
header_quote: "lovelace"
---

http://handong1587.github.io/deep_learning/2015/10/09/rnn-and-lstm.html

http://uploads1.wikiart.org/images/m-c-escher/ascending-descending.jpg

chris olah's postt on attention

**[quote: RNN bot trained on this text - ml4a.github.io -> link to torch-rnn code ]**

Although convolutional neural networks stole the spotlight with recent successes in image processing and eye-catching applications, in many ways recurrent neural networks (RNNs) are the variety of neural nets which are the most dynamic and exciting within the research community. This is because they make a critical innovation which dramatically extends the range of possible applications of neural nets: they operate over _sequences of data_.

This is useful because many, or even most problems in AI are sequential in nature. For instance, vision is not simply a function of what our eyes are seeing at that moment, it's a coming from a continuous stream of inputs over time, building up a mental depiction of a place. This is why one doesn't lapse their vision at every blink.

Additionally, RNNs are able to operate over sequences of data that are not fixed in length. This is another crucial advantage over feedforward neural nets whose inputs and outputs must have a fixed size.

 
## From feedforward to recurrent

RNNs share much in common with ordinary neural nets and convnets, and we can bootstrap our understanding of them from these similarities. Like those, RNNs possess an internal state which processes and transforms inputs into outputs, trained by a dataset to maximize the predictive accuracy between them. 

Let's call this internal (or "hidden") state $$H$$. In the kinds of neural nets we've seen before, the hidden state is tuned through the process of training, after which the internal weights and biases are fixed. This means they are static -- same input makes same output. In recurrent neural networks, the hidden state is not static -- it is a function of time (?). The way this is achieved is through the process of _recurrence_. where the hidden state is a function of the input _and_ the previous hidden state.

**[Figure: X->H->Y, X(t)->H(t)->Y(t)]**

But unlike feedforward neural nets, recurrent neural nets have a _hidden state_, $$h(t)$$, 

## The simplest kind of recurrent neural network

h(t) = Wx * x + Wh * h(t-1)
y = tanh(Ww * h(t))

## Processing sequences


## Architecture


etc
https://twitter.com/robinsloan/status/725068953383362560


## Etc

Robin Sloan robinsloan-lstm-author.mp4

LSTM vis http://lstm.seas.harvard.edu/

Olah Attention + augmented rnn http://distill.pub/2016/augmented-rnns/

Graves RNN class + RNN hallucinations https://www.youtube.com/watch?v=-yX1SYeDHbg&feature=youtu.be&t=41m50s

A long list of links to tutorials, code, and resources for using RNNs and LSTMs
http://handong1587.github.io/deep_learning/2015/10/09/rnn-and-lstm.html

RNNs in Tensorflow http://www.wildml.com/2016/08/rnns-in-tensorflow-a-practical-guide-and-undocumented-features/

Language modeling a billion words http://torch.ch/blog/2016/07/25/nce.html

http://deeplearning4j.org/lstm.html
http://nikhilbuduma.com/2015/01/11/a-deep-dive-into-recurrent-neural-networks/


image captioning for mortals https://indico.io/blog/neural-image-captioning-for-mortals/
https://github.com/karpathy/neuraltalk2
densecap http://cs.stanford.edu/people/karpathy/deepimagesent/

DRAW (nice image) https://github.com/jbornschein/draw
https://github.com/skaae/lasagne-draw

teaching RNNs about monet http://blog.manugarri.com/teaching-recurrent-neural-networks-about-monet/

neural-storyteller:
https://github.com/ryankiros/skip-thoughts
https://github.com/ryankiros/neural-storyteller 
https://medium.com/@samim/generating-stories-about-images-d163ba41e4ed#.isfi2qr5f

WORD-RNN https://github.com/larspars/word-rnn

karpathy visualizing and understanding RNN https://skillsmatter.com/skillscasts/6611-visualizing-and-understanding-recurrent-networks

RNN tutorial http://www.wildml.com/2015/10/recurrent-neural-networks-tutorial-part-3-backpropagation-through-time-and-vanishing-gradients/

colah understanding lstm http://colah.github.io/posts/2015-08-Understanding-LSTMs/

Anyone Can Learn To Code an LSTM-RNN in Python (Part 1: RNN) https://iamtrask.github.io/2015/11/15/anyone-can-code-lstm/

semantic object parsing https://twitter.com/evolvingstuff/status/713149843481317376
https://twitter.com/alexjc/status/716549734371102720/photo/1

https://handong1587.github.io/deep_learning/2015/10/09/rnn-and-lstm.html

lstm explained http://apaszke.github.io/lstm-explained.html
deep dive into RNN nikhilbuduma.com/2015/01/11/a-deep-dive-into-recurrent-neural-networks/

generative choreography http://peltarion.com/creative-ai

RNN + super mario https://medium.com/@ageitgey/machine-learning-is-fun-part-2-a26a10b68df3#.j4sxifd25

next gen text
https://twitter.com/robinsloan/status/832710183432237056

https://metamind.io/research/learning-when-to-skim-and-when-to-read

http://www.topbots.com/exploring-lstm-tutorial-part-1-recurrent-neural-network-deep-learning/
https://medium.com/towards-data-science/memory-attention-sequences-37456d271992


densecap.png
neural-storyteller.jpg
neural-storyteller2.png
samim-neural-storyteller.jpg

Understanding Hidden Memories of Recurrent Neural Networks https://arxiv.org/pdf/1710.10777.pdf