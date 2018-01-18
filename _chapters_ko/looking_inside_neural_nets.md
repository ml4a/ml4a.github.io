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

## 이런 또 실수를 했네

때로는 슬프게도 네트워크가 실수를 할 것입니다. 제가 보기에는 아래 첫 번째 이미지가 9인지 확실하지가 않습니다. 이 신경망처럼 누구든지 쉽게 4라고 생각할 수 있습니다. 비슷하게 신경망이 8이라고 잘못 분류한 두 번째 숫자가 왜 3인지 누군가는 이해할 수도 있습니다. 세 번째와 네 번째 숫자의 실수는 더 특이합니다. 거의 모든 사람이 이 숫자를 3과 2라고 즉시 인식할 것입니다. 하지만 시스템은 각각 이를 5라고 잘못 해석했고 그 다음은 거의 마땅한 근거를 찾지 못한 것 같습니다.

{% include figure_multi.md path1="/images/figures/mnist-mistakes.png" caption1="단일 층 MNIST 네트워크에 의해 잘못 분류된 예시. 왼쪽의 두 개는 이해가 되지만 오른쪽의 두개는 확실히 에러로 보입니다." %}

이전 장에서 MNIST 숫자 데이터에서 90% 정확도를 달성했던 마지막 신경망의 성능을 조금 더 자세히 살펴 보겠습니다. 이를 위한 한가지 방법은 예측 결과를 쪼개어 테이블에 나누어 놓은 오차 행렬(confusion matrix)를 사용하는 것입니다. 다음 오차 행렬에서 10개의 행은 MNIST 데이터셋의 실제 레이블에 해당하고, 열은 예측한 레이블을 나타냅니다. 예를 들어, 4번째 행과 6번째 열은 실제 3인 샘플 71개가 신경망에 의해 5로 잘못 분류되었다는 것을 보여줍니다. 오차 행렬의 녹색 대각선은 올바른 예측의 양을 보여 주고, 다른 모든 셀은 잘못된 예측입니다.

각 셀에 마우스를 올리면 각 셀에서 신경망의 예측 신뢰도(확률) 순으로 가장 높은 샘플을 보여줍니다.

{% include demo_insert.html path="/demos/confusion_mnist/" parent_div="post" %}

또한 아래처럼 오차 행렬의 각 셀의 최상위 샘플을 같이 그려서 유용한 통찰을 얻을 수 있습니다.

{% include figure_multi.md path1="/images/figures/mnist-confusion-samples.png" caption1="MNIST 오차 행렬에서 가장 신뢰도가 높은 샘플" %}

이 그림은 네트워크가 특정 종류의 예측을 어떻게 만드는 지에 대한 정보를 제공합니다. 이 네트워크는 큰 동그라미를 0으로 예측하는 것으로 보입니다. 처음 두 개의 열을 보면, 0을 예측하기 위해 큰 동그라미를 찾고, 1을 예측하기 위해서는 가느다란 직선을 찾는 것으로 보입니다. 그래서 이런 특징이 있는 다른 숫자들을 잘못 분류하고 있습니다.

## 신경망 내부를 들여다 보기

지금까지 손글씨 숫자를 인식하기 위해 훈련한 신경망을 보았습니다. 이 모델은 매우 인상적이지만 매우 쉬운 데이터셋입니다. 10개의 클래스가 비교적 잘 구성되어 있고 클래스가 같은 샘플들은 서로 많이 다르지 않아 모델을 만들기 편리합니다. 대부분의 실제 문제에서는 훨씬 이상적이지 않은 이미지들을 분류해야 합니다. 같은 신경망을 사용해 다른 데이터셋에서 성능을 측정해 보겠습니다. [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)는 32x32 크기의 레이블된 60,000개의 컬러 이미지입니다. 이 데이터셋에는 비행기, 자동차, 새, 고양이, 사슴, 강아지, 개구리, 말, 배, 트럭의 10개 클래스가 있습니다. 다음은 CIFAR-10에서 무작위로 선택한 샘플입니다.

{% include figure_multi.md path1="/images/figures/cifar-grid.png" caption1="CIFAR-10 이미지에서 무작위로 선택한 샘플" %}

이 이미지 클래스들은 이제까지 우리가 다뤄보지 못했던 종류임을 즉각적으로 알 수 있습니다. 예를 들어, 고양이는 다른 방향의 얼굴을 하고 있을 수 있고 색깔과 털의 패턴이 다를 수 있습니다. 또 몸을 쭉 펴고 있거나 웅크리고 있을 수 있는 등 손글씨 숫자에서는 보지 못했던 많은 변종이 있습니다. 고양이 사진은 다른 물체와 함께 있어서 문제를 더 어렵게 만듭니다.

당연하게 이런 이미지에 2층 신경망을 훈려시키면 정확도는 겨우 37% 정도에 이를 것입니다. 그래도 무작위로 추측하는 것(10%의 정확도)보다는 많이 낫지만 MNIST 분류기가 달성한 90%에는 훨씬 못 미칩니다. 합성곱 신경망을 사용하면 MNIST와 CIFAR-10 양쪽 모두에서 이 숫자는 크게 향상될 것입니다. 지금은 평범한 신경망의 가중치를 조사해서 이 방식의 단점에 대해 조금 더 알아 보도록 하겠습니다.

앞서 은닉층이 없는 1층 신경망의 가중치를 관찰하는 실험을 다시 하는데 이번에는 CIFAR-10의 이미지를 훈련시켜 보겠습니다. 가중치는 아래와 같이 보입니다.

{% include figure_multi.md path1="/images/figures/rolled_weights_cifar.png" caption1="1층 CIFAR-10 분류기의 가중치 시각화" %}

MNIST 가중치와 비교해 보면 거의 뚜렷한 특성이 없고 뭐라고 정의하기가 어렵습니다. 조금 자세히 살펴보면 일부 직관을 얻을 수 있습니다. 가령, 비행기와 배는 이미지 가장자리는 거의 푸른색이어서, 파란 하늘과 바다로 둘러 싸인 이미지들의 경향을 반영하고 있습니다. 특정 클래스에 대한 가중치 이미지는 그 클래스에 속한 이미지의 평균에 관련되어 있기 때문에, 이전처럼 뚜렷하지 않은 평균적인 색깔을 예상할 수 있습니다. 하지만 CIFAR 클래스안의 이미지들은 거의 일관성이 없기 때문에, MNIST에서 보았던 것처럼 잘 정의된 "원형"을 찾기가 어렵습니다.

이 CIFAR-10 분류기의 오차 행렬을 살펴 보겠습니다.

{% include demo_insert.html path="/demos/confusion_cifar/" parent_div="post" %}

예상대로 성능은 겨우 37% 정확도라서 매우 나쁩니다. 확실하게 간단한 1층 신경망은 이 데이터셋의 복잡한 특징을 잡아낼 능력이 없습니다. 성능을 조금 향상시킬 수 있는 한가지 방법은 은닉층을 추가하는 것입니다. 다음 절에서 이에 대한 효과를 확인해 보겠습니다.

## 은닉층 추가

지금까지 입력이 출력에 바로 연결되는 1층 신경망만 다루었습니다. 은닉층은 이 신경망에 어떤 영향을 미칠까요? 이를 알아 보기 위해 MNIST 네트워크에 10개의 뉴런을 가진 중간층을 삽입해 보겠습니다. 이제 손글씨 숫자를 분류하기 위한 신경망은 다음과 같아 집니다.

{% include figure_multi.md path1="/images/figures/mnist_2layers.png" caption1="MNIST를 위한 2층 신경망" %}

앞서 1층 신경망에서의 간단한 템플릿 비유는 이 경우에 적용되지 않습니다. 왜냐하면 784개의 입력 픽셀이 출력 클래스에 직접 연결되어 있지 않기 때문입니다. 어떤 의미에서 처음 만들었던 1층 신경망은 각각의 가중치가 하나의 클래스 레이블에 직접 연결되어 있어서 그 클래스에만 영향을 미치기 때문에 템플릿을 학습했다고 억지로 말할 수 있습니다. 하지만 여기에서처럼 더 복잡한 네트워크에서는 은닉층의 가중치는 출력층의 10개의 뉴런 모두에게 영향을 미칩니다. 그렇다면 이 가중치가 어떻게 나타날까요?

어떤 일이 일어나는지 이해하기 위해서 이전처럼 첫 번째 층에 있는 가중치를 시각화하겠습니다. 하지만 활성화 값이 클래스 점수를 만들기 위해 두 번째 층과 어떻게 연결되어 있는지도 주의깊게 살필 것입니다. 이미지가 첫 번째 층의 어떤 필터에 크게 동조하면 그 뉴런의 활성화 값이 크게 출력된다는 것을 기억하세요. 그러므로 은닉층에 잇는 뉴런 10개는 이미지에 있는 10개의 특성의 존재 여부를 반영합니다. 출력층에서는 클래스 레이블에 상응하는 하나의 뉴런은 이전 은닉층의 10개 활성화 값의 가중치 합입니다. 다음 그림을 한번 보죠.

{% include demo_insert.html path="/demos/f_mnist_weights/" parent_div="post" %}

위에 그림으로 나타난 첫 번째 층의 가중치를 살펴 보죠. 이 그림은 더 이상 이미지 클래스의 템플릿처럼 보이지 않고 오히려 낯설어 보입니다. 어떤 것들은 숫자 비슷한 모양 같기도 하고 다른 것들에서는 숫자의 일부분이 보입니다. 반 원, 대각선, 동심원 등입니다.

필터 이미지 아래 행은 이미지 클래스마다 하나씩인 출력 뉴런에 대응됩니다. 막대 그래프는 은닉층에 있는 필터 10개의 활성화에 연결된 가중치의 크기입니다. 예를 들어 `0` 클래스는 (0가 나타날 가능성이 높은) 바깥쪽 테두리를 따라 높게 활성화된 필터를 선호합니다. 가운데 픽셀(일반적으로 0의 구멍에 해당하는 픽셀)이 높은 값을 가진 필터는 선호하지 않습니다. `1` 클래스는 이와 거의 정반대입니다. `1`을 쓸 때 수직선이 나타므로 가운데 픽셀이 강한 필터를 선호합니다.

이런 접근 방법의 장점은 유연성입니다. 각 클래스에는 해당 출력 뉴런을 자극하는 다양한 입력 패턴이 있습니다. 각 클래스는 이전 은닉층에서 여러가지 추상적인 특성이나 특성들의 조합이 감지되면 활성화될 수 있습니다. 근본적으로 여러  종류의 0, 여러 종류의 1 등에 대해 학습할 수 있습니다. 이는 보통--항상은 아니고--대부분의 작업에서 네트워크의 성능을 향상시킵니다.

## Features and representations

Let's generalize some of what we've learned in this chapter. In single-layer and multi-layer neural networks, each layer has a similar function; it transforms data from the previous layer into a "higher-level" representation of that data. By "higher-level," we mean that it contains a compact and more salient representation of that data, in the way that a summary is a "high-level" representation of a book. For example, in the 2-layer network above, we mapped the "low-level" pixels into "higher-level" features found in digits (strokes, loops, etc) in the first layer, and then mapped those high-level features into an even higher-level representation in the next layer, that of the actual digits. This notion of transforming data into smaller but more meaningful information is at the heart of machine learning, and a primary capability of neural networks.

By adding a hidden layer into a neural network, we give it a chance to learn features at multiple levels of abstraction. This gives us a rich representation of the data, in which we have low-level features in the early layers, and high-level features in the later layers which are composed of the previous layers' features. 

As we saw, hidden layers can improve accuracy, but only to a limited extent. Adding more and more layers stops improving accuracy quickly, and comes at a computational cost -- we can't simply ask our neural network to memorize every possible version of an image class through its hidden layers. It turns out there is a better way, using [convolutional neural networks](/ml4a/convnets), which will be covered in a later chapter. 

# Further reading

{% include further_reading.md title="Demo: Tinker with a neural network" author="Daniel Smilkov and Shan Carter" link="http://playground.tensorflow.org" %} 

{% include further_reading.md title="Visualizing what ConvNets learn (Stanford CS231n)" author="Andrej Karpathy" link="https://cs231n.github.io/understanding-cnn/" %} 

## Next chapter

In the next chapter, we will learn about a critical topic that we've glossed over up until now, [how neural networks are trained](/ml4a/how_neural_networks_are_trained/): the process by which neural nets are constructed and trained on data, using a technique called gradient descent via backpropagation. We will build up our knowledge starting from simple linear regression, and working our way up through examples, and elaborating on the various aspects of training which researchers must deal with.