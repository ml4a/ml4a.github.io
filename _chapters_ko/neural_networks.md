---
layout: chapter
title: "Neural networks"
includes: [mathjax, jquery, convnetjs, dataset, convnet, visualizer]
header_image: "/images/headers/analytical_engine.jpg"
header_text: "“It were much to be desired, that when mathematical processes pass through the human brain instead of through the medium of inanimate mechanism, it were equally a necessity of things that the reasonings connected with operations should hold the same just place as a clear and well-defined branch of the subject of analysis, a fundamental but yet independent ingredient in the science, which they must do in studying the engine.” <a href=\"https://books.google.de/books?id=b8YUDAAAQBAJ&pg=PA16&lpg=PA16\">Sketch of the Analytical Engine (1843), Ada Lovelace</a>"
---

[中文](/ml4a/cn/neural_networks/)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[español](/ml4a/es/neural_networks/)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[한국어](/ml4a/ko/neural_networks/)

신경망이 처음 고안된 것은 거의 100여년 전으로 [에이다 러브레이스](http://findingada.com/)는 "[신경 시스템에 대한 수학 모델](http://www.thelancet.com/journals/lancet/article/PIIS0140-6736(15)00686-8/fulltext?rss=yes)"을 만들고자 하는 꿈을 가졌습니다. 뇌와 기계 사이의 추측성 비유는 계산 자체의 역사만큼이나 오래되었지만, 에이다의 선생님이었던 [찰스 배비지](https://ko.wikipedia.org/wiki/%EC%B0%B0%EC%8A%A4_%EB%B0%B0%EB%B9%84%EC%A7%80)가 [해석 기관](https://ko.wikipedia.org/wiki/%ED%95%B4%EC%84%9D%EA%B8%B0%EA%B4%80)을 제안하기 전까지는 "계산기"를 사람같은 인지 능력을 가진 것으로 상상하지는 못했습니다. 그 당시 기술자들은 그녀가 고안한 복잡한 회로를 만들 능력이 없었기 때문에 에이다는 생전에 이 기관이 실제로 구현되는 것을 보지 못했습니다. 그럼에도 불구하고, 이 아이디어는 다음 세기를 거쳐 전달되었고 [앨런 튜링](https://ko.wikipedia.org/wiki/%EC%95%A8%EB%9F%B0_%ED%8A%9C%EB%A7%81)이 "[튜링 테스트](https://en.wikipedia.org/wiki/Turing_test)"이라고 불리게 된 [모방 게임](http://phil415.pbworks.com/f/TuringComputing.pdf)을 소개할 때 여기에서 영감을 받았다고 인용하였습니다. 극단적으로 단순한 계산에 대한 그의 고찰은 첫 번째 인공 지능의 붐을 촉발시켰고 신경망의 첫 번째 전성기를 마련하였습니다.

## 신경망의 탄생과 재탄생

최근 신경망의 부활은 독특한 스토리를 가집니다. 초기 AI에 밀접하게 연관된 신경망은 1940년대 후반에 튜링의 [B타입 기계](https://en.wikipedia.org/wiki/Unorganized_machine) 형식으로 처음 형태를 갖추었고, 인간의 학습 과정을 연구하는 신경 과학자와 인지 신경학자들의 [신경가소성](https://en.wikipedia.org/wiki/Hebbian_theory)에 대한 초기 연구에 의존했습니다. 뇌의 발전 메커니즘이 밝혀짐에 따라 컴퓨터 과학자들은 기계에서 이 과정을 시뮬레이션하기 위해 활동 전위와 신경 역전파를 이상화한 모델을 실험했습니다.

오늘날 대부분의 과학자들은 너무 심각하게 이와 같이 비유하는 것을 경계합니다. 왜냐하면 신경망은 뇌를 정확히 묘사하기 위한 것이 아니라 머신러닝 문제를 해결하기 위해서만 고안되었기 때문입니다. 반면 완전히 다른 분야인 [계산 신경과학](https://en.wikipedia.org/wiki/Computational_neuroscience)은 뇌를 정확히 모델링하는 도전을 지속하고 있습니다. 그럼에도 불구하고, 신경망의 핵심 유닛을 단순화된 생물학적 뉴런으로 비유하는 것이 수십년 동안 계속되었습니다. 생물학적 뉴런에서 인공 뉴런으로의 변화는 다음 그림으로 요약할 수 있습니다.

{% include figure_multi.md path1="/images/neuron-anatomy.jpg"
caption1="Anatomy of a biological neuron<br/>Source: <a href=\"https://askabiologist.asu.edu/neuron-anatomy\">ASU school of life sciences</a>" path2="/images/neuron-simple.jpg"
caption2="Simplified neuron body within a network<br/>Source: <a href=\"http://www.generation5.org/content/2000/nn00.asp\">Gurney, 1997. An Introduction to Neural Networks</a>" path3="/images/figures/neuron.png" caption3="Artificial neuron<br/>&nbsp;" %}

 1950년대 후반에 [프랭크 로젠블라트](https://en.wikipedia.org/wiki/Frank_Rosenblatt)가 [이전 장](/ml4a/machine_learning/)에서 보았던 선형 분류기의 한 종류인 [퍼셉트론](https://ko.wikipedia.org/wiki/%ED%8D%BC%EC%85%89%ED%8A%B8%EB%A1%A0)을 고안하여 신경망의 발전에 큰 진전을 이루었습니다. 미국 해군으로 부터 재정을 지원받아 Mark 1 퍼셉트론이 광전지, 전위차계, 전기 모터를 사용해 이미지 인식을 수행하도록 설계되었습니다. 복잡한 전기 회로에서 얻은 효과를 보고 1958년 뉴욕 타임즈는 기계가 곧 ["걷고, 말하고, 보고, 쓰고, 스스로 재생산하며 자신의 존재를 인지할"](http://query.nytimes.com/gst/abstract.html?res=9D01E4D8173DE53BBC4053DFB1668383649EDE) 것이라고 예상했습니다.

이런 초기의 높은 인기는 공상 과학 소설가들에게 수십년간 영감을 주었지만, 학계안에서는 흥분이 많이 사그라들었습니다. 마빈 민스키와 시모어 페퍼트의 1969년 책 [퍼셉트론](https://en.wikipedia.org/wiki/Perceptrons_(book))에서 여러가지--[심지어 아주 간단한](http://users.ecs.soton.ac.uk/harnad/Hypermail/Explaining.Mind96/0140.html)--제약 사항을 시연했습니다. 이는 무심코 컴퓨터가 [엄청난 속도](https://ko.wikipedia.org/wiki/%EB%AC%B4%EC%96%B4%EC%9D%98_%EB%B2%95%EC%B9%99)의 연산 능력으로 게속 발전할 것이라고 잘 못 가정한 학계와 일반 대중 모두의 [관심을 낮추게](https://en.wikipedia.org/wiki/AI_winter) 하였습니다. 튜링 조차도 기계가 [Y2K 문제](https://ko.wikipedia.org/wiki/2000%EB%85%84_%EB%AC%B8%EC%A0%9C)가 있었던 2000년에는 인간 수준의 지능을 가질 것이라고 말했습니다.

80년대와 90년대에 조용하지만 여러가지 놀라운 발전에도 불구하고 [[1]](http://www.iro.umontreal.ca/~vincentp/ift3395/lectures/backprop_old.pdf)[[2]](http://yann.lecun.org/exdb/publis/pdf/lecun-89e.pdf)[[3]](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf), 2000년대 까지는 비주류였고 대부분의 상용이나 산업용 애플리케이션에는 [서포트 벡터 머신](https://ko.wikipedia.org/wiki/%EC%84%9C%ED%8F%AC%ED%8A%B8_%EB%B2%A1%ED%84%B0_%EB%A8%B8%EC%8B%A0)이나 다른 알고리즘들이 선호되었습니다. [2009년에 시작해서](http://www.cs.utoronto.ca/~gdahl/papers/dbnPhoneRec.pdf) 특히 [2012년부터 크게 성장하면서](https://www.technologyreview.com/s/530561/the-revolutionary-technique-that-quietly-changed-machine-vision-forever/) 신경망은 다시 한번 ML 알고리즘을 압도하게 되었습니다. 이들의 부활은 시청각 분야의 주요 문제에 대해 이전의 최고 성능을 (이따금 아주 놀라운 수준으로) 능가하는 [합성곱 신경망](/ml4a/convnets.html)과 [순환 신경망](/ml4a/RNNs.html)의 탄생에 의해 크게 도움 받았습니다. 하지만 더 흥미로운 것은, 이전에 없던 새로운 애플리케이션과 특히 예술가들이나 AI 분야 밖의 사람들의 흥미를 끄는 특징을 가지고 있다는 것입니다. 이 책은 지금 부터 몇 장에 걸쳐 특별히 합성곱 신경망에 대해 자세히 살펴 보겠습니다.

많은 학습 알고리즘들이 수년간 제안되었지만 신경망에 대부분 촛점을 맞추도록 하겠습니다. 왜냐하면:

 - 아주 간단하고 직관적인 공식으로 표현됩니다.
 - 심층 신경망은 이 책과 관련이 많은 중요한 여러가지 머신러닝 문제에서 최고의 성능을 냅니다.
 - 최근 머신러닝을 예술 분야에 접목한 것은 대부분 신경망을 사용했습니다.

## From linear classifiers to neurons

Recall from the previous chapter that the input to a 2d linear classifier or regressor has the form:

$$
\begin{eqnarray}
f(x_1, x_2) = b + w_1 x_1 + w_2 x_2
\end{eqnarray}
$$

More generally, in any number of dimensions, it can be expressed as

$$
\begin{eqnarray}
f(X) = b + \sum_i w_i x_i
\end{eqnarray}
$$

In the case of regression, $$f(X)$$ gives us our predicted output, given the input vector $$X$$. In the case of classification, our predicted class is given by

$$
\begin{eqnarray}
  \mbox{classification} & = & \left\{ \begin{array}{ll}
  	  1 & \mbox{if } f(X) \gt 0 \\
      0 & \mbox{if } f(X) \leq 0
      \end{array} \right.
\tag{1}\end{eqnarray}
$$

Each weight, $$w_i$$, can be interpreted as signifying the relative influence of the input that it's multiplied by, $$x_i$$. The $$b$$ term in the equation is often called the bias, because it controls how predisposed the neuron is to firing a 1 or 0, irrespective of the weights. A high bias makes the neuron require a larger input to output a 1, and a lower one makes it easier.

We can get from this formula to a full-fledged neural network by introducing two innovations. The first is the addition of an activation function, which turns our linear discriminator into what's called a neuron, or a "unit" (to dissociate them from the brain analogy). The second innovation is an architecture of neurons which are connected sequentially in layers. We will introduce these innovations in that order.

## Activation function

In both artificial and biological neural networks, a neuron does not just output the bare input it receives. Instead, there is one more step, called an activation function, analagous to the rate of [action potential](https://en.wikipedia.org/wiki/Action_potential) firing in the brain. The activation function takes the same weighted sum input from before, $$z = b + \sum_i w_i x_i$$, and then transforms it once more before finally outputting it.

Many activation functions have been proposed, but for now we will describe two in detail: sigmoid and ReLU.

Historically, the [sigmoid](https://en.wikipedia.org/wiki/Sigmoid_function) function is the oldest and most popular activation function. It is defined as:

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

$$e$$ denotes the [exponential constant](https://en.wikipedia.org/wiki/E_(mathematical_constant)), roughly equal to 2.71828. A neuron which uses a sigmoid as its activation function is called a sigmoid neuron. We first set the variable $$z$$ to our original weighted sum input, and then pass that through the sigmoid function.

$$
z = b + \sum_i w_i x_i \\
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

At first, this equation may seem complicated and arbitrary, but it actually has a very simple shape, which we can see if we plot the value of $$\sigma(z)$$ as a function of the input $$z$$.

{% include figure_multi.md path1="/images/figures/sigmoid.png" caption1="Sigmoid activation function" %}

We can see that $$\sigma(z)$$ acts as a sort of "squashing" function, condensing our previously unbounded output to the range 0 to 1. In the center, where $$z = 0$$, $$\sigma(0) = 1/(1+e^{0}) = 1/2$$. For large negative values of $$z$$, the $$e^{-z}$$ term in the denominator grows exponentially, and $$\sigma(z)$$ approaches 0. Conversely, large positive values of $$z$$ shrink $$e^{-z}$$ to 0, so $$\sigma(z)$$ approaches 1.

The sigmoid function is continuously differentiable, and its derivative, conveniently, is $$\sigma^\prime(z) = \sigma(z) (1-\sigma(z))$$. This is important because we have to use calculus to train neural networks, but don't worry about that for now.

Sigmoid neurons were the basis of most neural networks for decades, but in recent years, they have fallen out of favor. The reason for this will be explained in more detail later, but in short, they make neural networks that have many layers difficult to train due to the [vanishing gradient problem](https://en.wikipedia.org/wiki/Vanishing_gradient_problem). Instead, most have shifted to using another type of activation function, the rectified linear unit, or ReLU for short. Despite its obtuse name, it is simply defined as $$R(z) = max(0, z)$$.

{% include figure_multi.md path1="/images/figures/relu.png" caption1="ReLU activation function" %}

In other words, ReLUs let all positive values pass through unchanged, but just sets any negative value to 0. Although newer activation functions are gaining traction, most deep neural networks these days use ReLU or one of its [closely related variants](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)).

Regardless of which activation function is used, we can visualize a single neuron with this standard diagram, giving us a nice intuitive visual representation of a neuron's behavior.

{% include figure_multi.md path1="/images/figures/neuron.png" caption1="An artificial neuron" %}

The above diagram shows a neuron with three inputs, and outputs a single value $$y$$. As before, we first compute the weighted sum of its inputs, then pass it through an activation function $$\sigma$$.

$$
z = b + w_1 x_1 + w_2 x_2 + w_3 x_3 \\
y = \sigma(z)
$$

You may be wondering what the purpose of an activation function is, and why it is preferred to simply outputting the weighted sum, as we do with the linear classifier from the last chapter. The reason is that the weighted sum, $$z$$, is [_linear_](https://en.wikipedia.org/wiki/Linearity) with respect to its inputs, i.e. it has a flat dependence on each of the inputs. In contrast, non-linear activation functions greatly expand our capacity to model curved or otherwise non-trivial functions. This will become clearer in the next section.

# Layers

Now that we have described neurons, we can now define neural networks. A neural network is composed of a series of layers of neurons, such that all the neurons in each layer connect to the neurons in the next layer.

{% include figure_multi.md path1="/images/figures/neural-net.png" caption1="A 2-layer neural network" %}

Note that when we count the number of layers in a neural network, we only count the layers with connections flowing into them (omitting our first, or input layer). So the above figure is of a 2-layer neural network with 1 hidden layer. It contains 3 input neurons, 2 neurons in its hidden layer, and 1 output neuron.

Our computation starts with the input layer on the left, from which we pass the values to the hidden layer, and then in turn, the hidden layer will send its output values to the last layer, which contains our final value.

Note that it may look like the three input neurons send out multiple values because each of them are connected to both of the neurons in the hidden layer. But really there is still only one output value per neuron, it just gets copied along each of its output connections. Neurons always output one value, no matter how many subsequent neurons it sends it to.

# Regression

The process of a neural network sending an initial input forward through its layers to the output is called forward propagation or a forward pass and any neural network which works this way is called a feedforward neural network. As we shall soon see, there are some neural networks which allow data to flow in circles, but let's not get ahead of ourselves yet...

Let's demonstrate a forward pass with this interactive demo. Click the 'Next' button in the top-right corner to proceed. 

{% include demo_insert.html path="/demos/simple_forward_pass/" parent_div="post" %}

# More layers, more expressiveness

Why are hidden layers useful? The reason is that if we have no hidden layers and map directly from inputs to output, each input's contribution on the output is independent of the other inputs. In real-world problems, input variables tend to be highly interdependent and they affect the output in combinatorially intricate ways. The hidden layer neurons allow us to capture subtle interactions among our inputs which affect the final output downstream.
Another way to interpret this is that the hidden layers represent higher-level "features" or attributes of our data. Each of the neurons in the hidden layer weigh the inputs differently, learning some different intermediary characteristic of the data, and our output neuron is then a function of these instead of the raw inputs. By including more than one hidden layer, we give the network an opportunity to learn multiple levels of abstraction of the original input data before arriving at a final output. This notion of high-level features will become more concrete [in the next chapter](/ml4a/looking_inside_neural_nets/) when we look closely at the hidden layers.

Recall also that activation functions expand our capacity to capture non-linear relationships between inputs and outputs. By chaining multiple non-linear transformations together through layers, this dramatically increases the flexibility and expressiveness of neural networks. The proof of this is complex and beyond the scope of this book, but it can even be shown that any 2-layer neural network with a non-linear activation function (including sigmoid or ReLU) and enough hidden units is a [universal function approximator](http://www.sciencedirect.com/science/article/pii/0893608089900208), that is it's theoretically capable of expressing any arbitrary input-to-output mapping. This property is what makes neural networks so powerful.

# Classification

What about classification? In the previous chapter, we introduced binary classification by simply thresholding the output at 0; If our output was positive, we'd classify positively, and if it was negative, we'd classify negatively. For neural networks, it would be reasonable to adapt this approach for the final neuron, and classify positively if the output neuron scores above some threshold. For example, we can threshold at 0.5 for sigmoid neurons which are always positive.

But what if we have multiple classes? One option might be to create intervals in the output neuron which correspond to each class, but this would be problematic for reasons that we will learn about when we look at [how neural networks are trained](/ml4a/how_neural_networks_are_trained/). Instead, neural networks are adapted for classification by having one output neuron for each class. We do a forward pass and our prediction is the class corresponding to the neuron which received the highest value. Let's have a look at an example.

# Classification of handwritten digits

Let's now tackle a real world example of classification using neural networks, the task of recognizing and labeling images of handwritten digits. We are going to use the [MNIST dataset](http://yann.lecun.com/exdb/mnist/), which contains 60,000 labeled images of handwritten digits sized 28x28 pixels, whose classification accuracy serves as a common benchmark in machine learning research. Below is a random sample of images found in the dataset.
	
{% include figure_multi.md path1="/images/figures/fig_mnist_groundtruth.png" caption1="A random sample of MNIST handwritten digits" %}

The way we setup a neural network to classify these images is by having the raw pixel values be our first layer inputs, and having 10 output classes, one for each of our digit classes from 0 to 9. Since they are grayscale images, each pixel has a brightness value between 0 (black) and 255 (white). All the MNIST images are 28x28, so they contain 784 pixels. We can unroll these into a single array of inputs, like in the following figure.

{% include figure_multi.md path1="/images/figures/mnist-input.png" caption1="How to input an image into a neural network" %}

The important thing to realize is that although this network seems a lot more imposing than our simple 3x2x1 network in the previous chapter, it works exactly as before, just with many more neurons. Each neuron in the first hidden layer receives all the inputs from the first layer. For the output layer, we'll now have _ten_ neurons rather than just one, with full connections between it and the hidden layer, as before. Each of the ten output neurons is assigned to one class label; the first one is for  the digit `0`, the second for `1`, and so on.

After the neural network has been trained -- something we'll talk about in more detail [in a future chapter](/ml4a/how_neural_networks_are_trained/) -- we can predict the digit associated with unknown samples by running them through the same network and observing the output values. The predicted digit is that whose output neuron has the highest value at the end. The following demo shows this in action; click "next" to flip through more predictions.

{% include demo_insert.html path="/demos/forward_pass_mnist/" parent_div="post" %}

# Further reading

{% include further_reading.md title="Neural Networks and Deep Learning" author="Michael Nielsen" link="http://neuralnetworksanddeeplearning.com/" %} 

{% include further_reading.md title="A 'Brief' History of Neural Nets and Deep Learning" author="Andrey Kurenkov" link="http://www.andreykurenkov.com/writing/a-brief-history-of-neural-nets-and-deep-learning/" %} 

{% include further_reading.md title="Demo: Visualization of artificial neural networks" author="Mostafa Abdelraouf" link="http://experiments.mostafa.io/public/ffbpann/" %} 

{% include further_reading.md title="Video: Neural Networks Demystified" author="Welch Labs" link="https://www.youtube.com/watch?v=bxe2T-V8XRs" %} 

## Next chapter

In the next chapter, [looking inside neural networks](/ml4a/looking_inside_neural_nets/), we will analyze the internal states of neural networks more closely, building up intuitions on what sorts of information they capture, as well as pointing out the flaws of basic neural nets, building up motivation for introducing more complex features such as convolutional layers to be explored in later chapters.