---
layout: term
title: "Neuron"
---

A _neuron_, sometimes called a _unit_ or _node_, is the building block of an [artificial neural network](/glossary/neural_network/). Neurons carry and forward a numeric value, sometimes called its _activation_, which is a function of the activations of a series of neurons which connect to it.

{:.center}
![neuron](/images/figures/neuron.png 'neuron')

For example, the above neuron carries a value $$y$$. To obtain the value of $$y$$, we first take a weighted sum of the activations of three neurons connecting to it, which we'll call $$z$$:

$$
\begin{eqnarray}
z = b + w_1 x_1 + w_2 x_2 + w_3 x_3
\end{eqnarray}
$$

$$z$$ is then further transformed by an _activation function_, like for example the sigmoid function $$\sigma(z) = \frac{1}{1 + e^{-z}}$$.

So the activation of this neuron would finally be equal to:

$$
y = \frac{1}{1 + e^{-(b + w_1 x_1 + w_2 x_2 + w_3 x_3)}}
$$

Neurons are introduced in the chapter on [neural networks](/ml4a/neural_networks/).