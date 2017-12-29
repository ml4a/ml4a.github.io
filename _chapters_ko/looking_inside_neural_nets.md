---
layout: chapter
title: "신경망의 내부"
includes: [mathjax, jquery, convnetjs, dataset, convnet, visualizer]
header_image: "/images/headers/brainbow.jpg"
header_text: "<a href=\"http://www.olympusbioscapes.com/gallery/images/743\">케이티 매소(Katie Matho) 박사</a>가 만든 출생 직후 생쥐의 브레인보우. <a href=\"https://en.wikipedia.org/wiki/Brainbow\">브레인보우</a>는 형광 단백질을 사용해 개별 뉴런을 시각화하는 뇌영상 기술입니다."
---

[English](/ml4a/looking_inside_neural_nets/)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[español](/ml4a/es/looking_inside_neural_nets/)

[이전 장](/ml4a/ko/neural_networks)에서 손글씨 숫자 이미지를 90%에 가까운 정확도로 분류하기 위해 신경망이 어떻게 훈련되는지 보았습니다. 이번 장에서는 이 성능을 조금 더 주의깊게 평가해 보고, 도대체 무슨 일이 일어나는지 직관을 얻기위해 내부 상태를 조사해 보려고 합니다. 이 장의 후반부에, 개, 자동차, 배 같은 물체가 있는 복잡한 데이터셋에서 신경망을 훈련시켜 보면서 문제점을 찾아 보겠습니다. 그리고 더 높은 수준에 도달하기 위해 어떤 혁신이 필요한지 추측해 보겠습니다.

## 가중치 시각화하기

MNIST 손글씨 숫자를 분류하는 네트워크를 훈련시켜 보겠습니다. 이전 장과는 다르게 은닉층 없이 입력층을 출력층에 바로 연결하겠습니다. 이 네트워크는 다음과 같을 것입니다.

{% include figure_multi.md path1="/images/figures/mnist_1layer.png" caption1="MNIST를 위한 단일층 신경망. 0에서부터 9까지 10개의 숫자 클래스에 대응하는 10개의 출력 뉴런이 있습니다." %}

이미지를 신경망에 입력할 때, 아래 그림의 왼쪽처럼 일렬로 늘어선 뉴런에 픽셀을 펼치는 모습으로 네트워크 그림을 그렸습니다. $$z$$로 표시된 첫 번째 출력 뉴런에 이어진 연결에만 초점을 맞추겠습니다. 입력 뉴런과 그에 상응하는 가중치는 각각 $$x_i$$와 $$w_i$$로 표시되어 있습니다.

{% include figure_multi.md path1="/images/figures/weights_analogy_1.png" caption1="출력 뉴런 하나의 가중치 연결" %}

하지만 픽셀을 펼치는 대신에, 가중치를 그에 상응하는 픽셀에 맞게 정렬된 28x28 격자로 볼 수 있습니다. 오른쪽 위의 표현이 아래 그림과 달라보이지만, 둘은 같은 공식, $$z=b+\sum{w x}$$으로 표현됩니다. 

{% include figure_multi.md path1="/images/figures/weights_analogy_2.png" caption1="출력 뉴런에 대한 픽셀-가중치 곱셉을 표현하는 다른 방법" %}

이제 이 구조로 학습된 신경망을 가지고, 숫자 0을 분류하는 책임을 가진 첫 번째 출력 뉴런에 연결된 가중치를 시각화해 보겠습니다. 낮은 가중치는 검게 그리고 높은 값은 흰색으로 표시합니다.

{% include figure_multi.md path1="/images/figures/rolled_weights_mnist_0.png" caption1="MNIST 분류기의 0-뉴런의 가중치 시각화" %}

눈을 조금 가늘게 뜨고 보면 약간 번진 0처럼 보이나요? 이렇게 보이는 이유는 뉴런이 하는 일을 생각해 보면 명확해 집니다. 이 뉴런은 0을 분류하는 책임을 가졌기 때문에, 0에 대해서 높은 값을 출력하고 0이 아닌 경우에는 낮은 값을 출력하는 것이 목적입니다. 일반적으로 0의 이미지에서 높은 값을 가지는 픽셀에 맞추어 큰 가중치를 가짐으로써 0에 대한 높은 출력을 만들 수 있습니다. 이와 동시에, 0이 아닌 이미지에서는 높고, 0인 이미지에서는 낮은 경향이 있는 픽셀에 있는 가중치는 작은 값을 가지게 되어 0이 아닐 때 비교적 낮은 출력을 얻을 수 있습니다. 가중치 이미지의 가운데 비교적 검은 부분은 0인 이미지가 이 부분에서는 비어있다는 사실을 의미합니다(0의 가운데 원부분). 하지만 보통 다른 숫자에서는 높게 나올 것입니다.

출력 뉴런 10개에서 학습된 가중치를 보묻 보겠습니다. 예상대로 열 개의 숫자가 모두 번진 것처럼 보입니다. 이들은 각 숫자 클래스에 속한 여러 이미지를 평균낸 것처럼 보입니다.

{% include figure_multi.md path1="/images/figures/rolled_weights_mnist.png" caption1="MNIST 분류기의 출력 뉴런의 가중치 시각화" %}

2에 대한 이미지를 입력으로 받았다고 가정해 보겠습니다. 그럼 2에 대한 분류를 담당하는 뉴런이 2가 나타날 가능성이 높은 픽셀을 따라 큰 가중치를 가지고 있기 때문에, 결국 이 뉴런이 높은 값을 가질 것으로 예상할 수 있습니다. 다른 뉴런에서도 가중치의 일부가 높은 값을 가진 픽셀과 맞을 수 있어 이 뉴런의 값들이 조금 높을 수 있습니다. 하지만 겹치는 부분이 훨씬 적고 이미지에 있는 여러 높은 값의 픽셀들이 2를 담당하는 뉴런에 있는 낮은 가중치 때문에 효과가 감쇠됩니다. 활성화 함수는 단조 함수이므로 이 결과를 바꾸지 않습니다. 즉 활성화 함수는 높은 입력을 받으면 높은 출력을 만듭니다.

이런 가중치를 출력 클래스의 템플릿을 구성하는 것으로 해석할 수 있습니다. 이는 매우 흥미로운데 왜냐하면 이런 숫자들이 무엇인지 어떤 의미인지 사전에 어떤 것도 네트워크에 알려 주지 않았지만 이런 이미지의 클래스를 닮게 되었기 때문입니다. 이것이 신경망의 내부에 관해 정말 흥미로운 것에 대한 힌트입니다. 신경망은 그들이 학습한 오브젝트의 _표현_ 을 형성합니다. 그리고 이런 표현은 간단한 분류나 예측보다 훨씬 더 유용할 수 있습니다. 이런 표현 능력은 [합성곱 신경망](/ml4a/convnets/)을 배울 때 다시 살펴 보고 여기서는 너무 깊이 다루지 않겠습니다.

해답 보다는 궁금증이 더 많이 생깁니다. 가령 은닉층을 추가할 때 가중치는 어떻게 될까요? 잠시 후에 보겠지만 이에 대한 답은 직관적으로 이전 절에서 보았던 것을 기초로 합니다. 하지만 이에 대한 답을 하기전에 신경망의 성능을 측정하는 것이 필요합니다. 특히 어떤 종류의 실수를 하게되는지 생각해 보겠습니다.

## 0op5, 1 d14 17 2ga1n

Occasionally, our network will make mistakes that we can sympathize with. To my eye, it's not obvious that the first digit below is 9. One could easily mistake it for a 4, as our network did. Similarly, one could understand why the second digit, a 3, was misclassified by the network as an 8. The mistakes on the third and fourth digits below are more glaring. Almost any person would immediately recognize them as a 3 and a 2, respectively, yet our machine misinterpreted the first as a 5, and is nearly clueless on the second.

{% include figure_multi.md path1="/images/figures/mnist-mistakes.png" caption1="A selection of mistakes by our 1-layer MNIST network. The two on the left are understandable; the two on the right are more obvious errors." %}

Let's look more closely at the performance of the last neural network of the previous chapter, which achieved 90% accuracy on MNIST digits. One way we can do this is by looking at a confusion matrix, a table which breaks down our predictions into a table. In the following confusion matrix, the 10 rows correspond to the actual labels of the MNIST dataset, and the columns represent the predicted labels. For example, the cell at the 4th row and 6th column shows us that there were 71 instances in which an actual 3 was mislabeled by our neural network as a 5. The green diagonal of our confusion matrix shows us the quantities of correct predictions, whereas every other cell shows mistakes.

Hover your mouse over each cell to get a sampling of the top instances from each cell, ordered by the network's confidence (probability) for the prediction.

{% include demo_insert.html path="/demos/confusion_mnist/" parent_div="post" %}

We can also get some nice insights by plotting the top sample for each cell of the confusion matrix, as seen below.

{% include figure_multi.md path1="/images/figures/mnist-confusion-samples.png" caption1="Top-confidence samples from an MNIST confusion matrix" %}

This gives us an impression of how the network learns to make certain kinds of predictions. Looking at first two columns, we see that our network appears to be looking for big loops to predict 0s, and thin lines to predict 1s, mistaking other digits if they happen to have those features.


## Breaking our neural network

So far we've looked only at neural networks trained to identify handwritten digits. This gives us many insights but is a very easy choice of dataset, giving us many advantages; We have only ten classes, which are very well-defined and have relatively little internal variance among them. In most real-world scenarios, we are trying to classify images under much less ideal circumstances. Let's look at the performance of the same neural network on another dataset, [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html), a labeled set of 60,000 32x32 color images belonging to ten classes: airplanes, automobiles, birds, cats, deer, dogs, frogs, horses, ships, and trucks. The following is a random sample of images from CIFAR-10.

{% include figure_multi.md path1="/images/figures/cifar-grid.png" caption1="A random sample from CIFAR-10 image set" %}

Right away, it's clear we must contend with the fact that these image classes differ in ways that we haven't dealt with yet. For example, cats can be facing different directions, have different colors and fur patterns, be outscretched or curled up, and many other variations we don't encounter with handwritten digits. Photos of cats will also be cluttered with other objects, further complicating the problem. 

Sure enough, if we train a 2-layer neural network on these images, our accuracy reaches only 37%. That's still much better than taking random guesses (which would get us a 10% accuracy) but it's far short of the 90% our MNIST classifier achieves. When we start convolutional neural networks, we'll improve greatly on those numbers, for both MNIST and CIFAR-10. For now, we can get a more precise sense about the shortcomings of ordinary neural networks by inspecting their weights.

Let's repeat the earlier experiment of observing the weights of a 1-layer neural network with no hidden layer, except this time training on images from CIFAR-10. The weights appear below.

{% include figure_multi.md path1="/images/figures/rolled_weights_cifar.png" caption1="Visualizing the weights for 1-layer CIFAR-10 classifier" %}

Compared to the MNIST weights, these have fewer obvious features and far less definition to them. Certain details do make intuitive sense, e.g. airplanes and ships are mostly blue on the outer edges of the images, reflecting the tendency for those images to have blue skies or waters around them. Because the weights image for a particular class does correlate to an average of images belonging to that class, we can expect blobby average colors to come out, as before. But because the CIFAR classes are much less internally consistent, the well-defined "templates" we saw with MNIST are far less evident.

Let's take a look at the confusion matrix associated with this CIFAR-10 classifier.

{% include demo_insert.html path="/demos/confusion_cifar/" parent_div="post" %}

Not surprisingly, its performance is very poor, reaching only 37% accuracy. Clearly, our simple 1-layer neural network is not capable of capturing the complexity of this dataset. One way we can improve its performance somewhat is by introducing a hidden layer. The next section will analyze the effects of doing that.

## Adding hidden layers

So far, we've focused on 1-layer neural networks where the inputs connect directly to the outputs. How do hidden layers affect our neural network? To see, let's try inserting a middle layer of ten neurons into our MNIST network. So now, our neural network for classifying handwritten digits looks like the following.

{% include figure_multi.md path1="/images/figures/mnist_2layers.png" caption1="2-layer neural network for MNIST" %}

Our simple template metaphor in the 1-layer network above doesn't apply to this case, because we no longer have the 784 input pixels connecting directly to the output classes. In some sense, you could say that we had "forced" our original 1-layer network to learn those templates because each of the weights connected directly into a single class label, and thus only affected that class. But in the more complicated network that we have introduced now, the weights in the hidden layer affect _all ten_ of the neurons in the output layer. So how should we expect those weights to look now?

To understand what's going on, we will visualize the weights in the first layer, as before, but we'll also look carefully at how their activations are then combined in the second layer to obtain class scores. Recall that an image will generate a high activation in a particular neuron in the first layer if the image is largely sympathetic to that filter. So the ten neurons in the hidden layer reflect the presence of those ten features in the original image. In the output layer, a single neuron, corresponding to a class label, is a weighted combination of those previous ten hidden activations. Let's look at them below.

{% include demo_insert.html path="/demos/f_mnist_weights/" parent_div="post" %}

Let's start with the first layer weights, visualized at the top. They don't look like the image class templates anymore, but rather more unfamiliar. Some look like pseudo-digits, and others appear to be components of digits: half loops, diagonal lines, holes, and so on.

The rows below the filter images correspond to our output neurons, one for each image class. The bars signify the weights associated to each of the ten filters' activations from the hidden layer. For example, the `0` class appears to favor first layer filters which are high along the outer rim (where a zero digit tends to appear). It disfavors filters where pixels in the middle are low (where the hole in zeros is usually found). The `1` class is almost the opposite of this, preferring filters which are strong in the middle, where you might expect the vertical stroke of a `1` to be drawn.

The advantage of this approach is flexibility. For each class, there is a wider array of input patterns that stimulate the corresponding output neuron. Each class can be triggered by the presence of several abstract features from the previous hidden layer, or some combination of them. Essentially, we can learn different kinds of zeros, different kinds of ones, and so on for each class. This will usually--but not always--improve the performance of the network for most tasks.

## Features and representations

Let's generalize some of what we've learned in this chapter. In single-layer and multi-layer neural networks, each layer has a similar function; it transforms data from the previous layer into a "higher-level" representation of that data. By "higher-level," we mean that it contains a compact and more salient representation of that data, in the way that a summary is a "high-level" representation of a book. For example, in the 2-layer network above, we mapped the "low-level" pixels into "higher-level" features found in digits (strokes, loops, etc) in the first layer, and then mapped those high-level features into an even higher-level representation in the next layer, that of the actual digits. This notion of transforming data into smaller but more meaningful information is at the heart of machine learning, and a primary capability of neural networks.

By adding a hidden layer into a neural network, we give it a chance to learn features at multiple levels of abstraction. This gives us a rich representation of the data, in which we have low-level features in the early layers, and high-level features in the later layers which are composed of the previous layers' features. 

As we saw, hidden layers can improve accuracy, but only to a limited extent. Adding more and more layers stops improving accuracy quickly, and comes at a computational cost -- we can't simply ask our neural network to memorize every possible version of an image class through its hidden layers. It turns out there is a better way, using [convolutional neural networks](/ml4a/convnets), which will be covered in a later chapter. 

# Further reading

{% include further_reading.md title="Demo: Tinker with a neural network" author="Daniel Smilkov and Shan Carter" link="http://playground.tensorflow.org" %} 

{% include further_reading.md title="Visualizing what ConvNets learn (Stanford CS231n)" author="Andrej Karpathy" link="https://cs231n.github.io/understanding-cnn/" %} 

## Next chapter

In the next chapter, we will learn about a critical topic that we've glossed over up until now, [how neural networks are trained](/ml4a/how_neural_networks_are_trained/): the process by which neural nets are constructed and trained on data, using a technique called gradient descent via backpropagation. We will build up our knowledge starting from simple linear regression, and working our way up through examples, and elaborating on the various aspects of training which researchers must deal with.