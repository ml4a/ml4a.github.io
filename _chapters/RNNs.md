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