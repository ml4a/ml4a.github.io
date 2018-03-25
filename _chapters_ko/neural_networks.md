---
layout: chapter
title: "신경망"
includes: [mathjax, jquery, convnetjs, dataset, convnet, visualizer]
header_image: "/images/headers/analytical_engine.jpg"
header_text: "“수학적인 처리 과정이 무생물인 장치가 아니라 사람의 뇌를 통과할 때, 연산에 연관된 추론이 명확히 잘 정의된 분석의 주체가 되는 것이 동일하게 필요합니다. 이는 과학에서 근본적이지만 아직 독립적인 주제로서 엔진 연구에 반드시 포함되어야 합니다.” <a href=\"https://books.google.de/books?id=b8YUDAAAQBAJ&pg=PA16&lpg=PA16\">해석 기관에 대한 스케치 (1843), 에이다 러브레이스</a>"
translator: "Haesun Park"
translator_link: "https://tensorflow.blog/"
---

[English](/ml4a/neural_networks/)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[中文](/ml4a/cn/neural_networks/)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[español](/ml4a/es/neural_networks/)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[français](/ml4a/fr/neural_networks/)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[日本語](/ml4a/jp/neural_networks/)

신경망(neural network)이 처음 고안된 것은 거의 100여년 전으로 [에이다 러브레이스](http://findingada.com/)는 "[신경 시스템에 대한 수학 모델](http://www.thelancet.com/journals/lancet/article/PIIS0140-6736(15)00686-8/fulltext?rss=yes)"을 만들고자 하는 꿈을 가졌습니다. 뇌와 기계 사이의 추측성 비유는 계산 자체의 역사만큼이나 오래되었지만, 에이다의 선생님이었던 [찰스 배비지](https://ko.wikipedia.org/wiki/%EC%B0%B0%EC%8A%A4_%EB%B0%B0%EB%B9%84%EC%A7%80)가 [해석 기관](https://ko.wikipedia.org/wiki/%ED%95%B4%EC%84%9D%EA%B8%B0%EA%B4%80)을 제안하기 전까지는 "계산기"를 사람같은 인지 능력을 가진 것으로 상상하지는 못했습니다. 그 당시 기술자들은 그녀가 고안한 복잡한 회로를 만들 능력이 없었기 때문에 에이다는 생전에 이 기관이 실제로 구현되는 것을 보지 못했습니다. 그럼에도 불구하고, 이 아이디어는 다음 세기를 거쳐 전달되었고 [앨런 튜링](https://ko.wikipedia.org/wiki/%EC%95%A8%EB%9F%B0_%ED%8A%9C%EB%A7%81)이 "[튜링 테스트](https://en.wikipedia.org/wiki/Turing_test)"이라고 불리게 된 [모방 게임](http://phil415.pbworks.com/f/TuringComputing.pdf)을 소개할 때 여기에서 영감을 받았다고 인용하였습니다. 극단적으로 단순한 계산에 대한 그의 고찰은 첫 번째 인공 지능의 붐을 촉발시켰고 신경망의 첫 번째 전성기를 마련하였습니다.

## 신경망의 탄생과 재탄생

최근 신경망의 부활은 독특한 스토리를 가집니다. 초기 AI에 밀접하게 연관된 신경망은 1940년대 후반에 튜링의 [B타입 기계](https://en.wikipedia.org/wiki/Unorganized_machine) 형식으로 처음 형태를 갖추었고, 인간의 학습 과정을 연구하는 신경 과학자와 인지 신경학자들의 [신경가소성](https://en.wikipedia.org/wiki/Hebbian_theory)에 대한 초기 연구에 의존했습니다. 뇌의 발전 메커니즘이 밝혀짐에 따라 컴퓨터 과학자들은 기계에서 이 과정을 시뮬레이션하기 위해 활동 전위와 신경 역전파를 이상화한 모델을 실험했습니다.

오늘날 대부분의 과학자들은 너무 심각하게 이와 같이 비유하는 것을 경계합니다. 왜냐하면 신경망은 뇌를 정확히 묘사하기 위한 것이 아니라 머신러닝 문제를 해결하기 위해서만 고안되었기 때문입니다. 반면 완전히 다른 분야인 [계산 신경과학](https://en.wikipedia.org/wiki/Computational_neuroscience)은 뇌를 정확히 모델링하는 도전을 지속하고 있습니다. 그럼에도 불구하고, 신경망의 핵심 유닛(unit)을 단순화된 생물학적 뉴런(neuron)으로 비유하는 것이 수십년 동안 계속되었습니다. 생물학적 뉴런에서 인공 뉴런으로의 변화는 다음 그림으로 요약할 수 있습니다.

{% include figure_multi.md path1="/images/neuron-anatomy.jpg"
caption1="생물학적 뉴런의 해부도<br/>출처: <a href=\"https://askabiologist.asu.edu/neuron-anatomy\">ASU school of life sciences</a>" path2="/images/neuron-simple.jpg"
caption2="네트워크에서의 단순화된 뉴런<br/>출처: <a href=\"http://www.generation5.org/content/2000/nn00.asp\">Gurney, 1997. An Introduction to Neural Networks</a>" path3="/images/figures/neuron.png" caption3="인공 뉴런<br/>&nbsp;" %}

 1950년대 후반에 [프랭크 로젠블라트](https://en.wikipedia.org/wiki/Frank_Rosenblatt)가 [이전 장](/ml4a/machine_learning/)에서 보았던 선형 분류기의 한 종류인 [퍼셉트론](https://ko.wikipedia.org/wiki/%ED%8D%BC%EC%85%89%ED%8A%B8%EB%A1%A0)을 고안하여 신경망의 발전에 큰 진전을 이루었습니다. 미국 해군으로 부터 재정을 지원받아 Mark 1 퍼셉트론이 광전지, 전위차계, 전기 모터를 사용해 이미지 인식을 수행하도록 설계되었습니다. 복잡한 전기 회로에서 얻은 효과를 보고 1958년 뉴욕 타임즈는 기계가 곧 ["걷고, 말하고, 보고, 쓰고, 스스로 재생산하며 자신의 존재를 인지할"](http://query.nytimes.com/gst/abstract.html?res=9D01E4D8173DE53BBC4053DFB1668383649EDE) 것이라고 예상했습니다.

이런 초기의 높은 인기는 공상 과학 소설가들에게 수십년간 영감을 주었지만, 학계안에서는 흥분이 많이 사그라들었습니다. 마빈 민스키와 시모어 페퍼트의 1969년 책 [퍼셉트론](https://en.wikipedia.org/wiki/Perceptrons_(book))에서 여러가지--[심지어 아주 간단한](http://users.ecs.soton.ac.uk/harnad/Hypermail/Explaining.Mind96/0140.html)--제약 사항을 시연했습니다. 이는 무심코 컴퓨터가 [엄청난 속도](https://ko.wikipedia.org/wiki/%EB%AC%B4%EC%96%B4%EC%9D%98_%EB%B2%95%EC%B9%99)의 연산 능력으로 게속 발전할 것이라고 잘 못 가정한 학계와 일반 대중 모두의 [관심을 낮추게](https://en.wikipedia.org/wiki/AI_winter) 하였습니다. 튜링 조차도 기계가 [Y2K 문제](https://ko.wikipedia.org/wiki/2000%EB%85%84_%EB%AC%B8%EC%A0%9C)가 있었던 2000년에는 인간 수준의 지능을 가질 것이라고 말했습니다.

80년대와 90년대에 조용하지만 여러가지 놀라운 발전에도 불구하고 [[1]](http://www.iro.umontreal.ca/~vincentp/ift3395/lectures/backprop_old.pdf)[[2]](http://yann.lecun.org/exdb/publis/pdf/lecun-89e.pdf)[[3]](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf), 2000년대 까지는 비주류였고 대부분의 상용이나 산업용 애플리케이션에는 [서포트 벡터 머신](https://ko.wikipedia.org/wiki/%EC%84%9C%ED%8F%AC%ED%8A%B8_%EB%B2%A1%ED%84%B0_%EB%A8%B8%EC%8B%A0)이나 다른 알고리즘들이 선호되었습니다. [2009년에 시작해서](http://www.cs.utoronto.ca/~gdahl/papers/dbnPhoneRec.pdf) 특히 [2012년부터 크게 성장하면서](https://www.technologyreview.com/s/530561/the-revolutionary-technique-that-quietly-changed-machine-vision-forever/) 신경망은 다시 한번 ML 알고리즘을 압도하게 되었습니다. 이들의 부활은 시청각 분야의 주요 문제에 대해 이전의 최고 성능을 (이따금 아주 놀라운 수준으로) 능가하는 [합성곱 신경망](/ml4a/convnets.html)과 [순환 신경망](/ml4a/RNNs.html)의 탄생에 의해 크게 도움 받았습니다. 하지만 더 흥미로운 것은, 이전에 없던 새로운 애플리케이션과 특히 예술가들이나 AI 분야 밖의 사람들의 흥미를 끄는 특징을 가지고 있다는 것입니다. 이 책은 지금 부터 몇 장에 걸쳐 특별히 합성곱 신경망에 대해 자세히 살펴 보겠습니다.

많은 학습 알고리즘들이 수년간 제안되었지만 신경망에 대부분 촛점을 맞추도록 하겠습니다. 왜냐하면:

 - 아주 간단하고 직관적인 공식으로 표현됩니다.
 - 심층 신경망은 이 책과 관련이 많은 중요한 여러가지 머신러닝 문제에서 최고의 성능을 냅니다.
 - 최근 머신러닝을 예술 분야에 접목한 것은 대부분 신경망을 사용했습니다.

## 선형 분류기에서 부터 뉴런까지

이전 장에서 배웠던 것을 다시 보면, 2d 선형 분류기나 회귀(regression) 모델로의 입력은 다음과 같은 형태를 가집니다:

$$
\begin{eqnarray}
f(x_1, x_2) = b + w_1 x_1 + w_2 x_2
\end{eqnarray}
$$

더 일반적으로 임의의 차원에 대해서는 다음과 같이 표현됩니다

$$
\begin{eqnarray}
f(X) = b + \sum_i w_i x_i
\end{eqnarray}
$$

회귀의 경우에는 입력 벡터 $$X$$가 주어졌을 때, $$f(X)$$가 예측 출력을 만듭니다. 분류(classification)에서는 예측 클래스가 다음과 같습니다.

$$
\begin{eqnarray}
  \mbox{분류} & = & \left\{ \begin{array}{ll}
  	  1 & \mbox{if } f(X) \gt 0 \\
      0 & \mbox{if } f(X) \leq 0
      \end{array} \right.
\tag{1}\end{eqnarray}
$$

$$x_i$$가 곱해지는 각 가중치 $$w_i$$는 입력의 상대적 영향을 의미하는 것으로 해석할 수 있습니다. 이 식에서 $$b$$ 항은 가중치와 상관없이 뉴런이 1 또는 0이 되는 성향을 제어하기 때문에 편향이라고 종종 부릅니다. 높은 편향은 뉴런이 출력 1을 만들기 위해 더 많은 입력을 필요로 한다는 의미이고 낮은 편향은 더 쉽게 출력 1을 만든다는 뜻입니다.

두 가지 혁신을 통해 이 공식으로 부터 완전한 신경망을 만들 수 있습니다. 첫 번째는 선형 식별기를 뉴런 또는 (뇌와 비유하지 않으려면) "유닛"으로 부르게 만든 활성화 함수(activation function)의 추가입니다. 두 번째 혁신은 층(layer)별로 순서대로 뉴런을 연결하는 구조입니다. 차례대로 이 혁신들을 소개하겠습니다.

## 활성화 함수

인공 신경망과 생물학적 신경망 모두 뉴런이 받은 입력을 그대로 출력하지 않습니다. 대신 뇌의 [활동 전위](https://ko.wikipedia.org/wiki/%ED%99%9C%EB%8F%99%EC%A0%84%EC%9C%84) 발화율에 대응하는 활성화 함수라고 부르는 한 가지 단계를 더 거칩니다. 활성화 함수는 이전 단계에서 나온 가중치 합, $$z = b + \sum_i w_i x_i$$을 입력으로 받아 최종적으로 출력하기 전에 이를 다시 한번 변형시킵니다.

많은 활성화 함수가 제안되었지만 여기서는 시그모이드(sigmoid)와 ReLU(rectified linear unit) 두 함수를 자세히 설명하겠습니다.

역사적으로 봤을 때, [시그모이드](https://en.wikipedia.org/wiki/Sigmoid_function) 함수는 가장 오래되고 또 널리 사용되는 활성화 함수입니다. 이 함수는 다음과 같이 정의됩니다:

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

$$e$$는 [자연 상수](https://ko.wikipedia.org/wiki/E_(%EC%83%81%EC%88%98))이며 대략 2.71828 입니다. 시그모이드 함수를 활성화 함수로 사용하는 뉴런을 시그모이드 뉴런이라고 부릅니다. 먼저 변수 $$z$$에 가중치 합을 입력으로 넣고, 그 다음 시그모이드 함수를 통과시킵니다.

$$
z = b + \sum_i w_i x_i \\
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

처음에는 이 식이 복잡하고 의미없어 보이지만, 사실 입력 $$z$$에 대한 함수로 $$\sigma(z)$$의 값을 그래프로 그리면 매우 간단한 형태를 가집니다.

{% include figure_multi.md path1="/images/figures/sigmoid.png" caption1="시그모이드 활성화 함수" %}

무한대의 출력을 0에서 1사이로 응축시키므로 $$\sigma(z)$$를 압축 함수의 하나로 볼 수 있습니다. $$z = 0$$인 가운데에서는 $$\sigma(0) = 1/(1+e^{0}) = 1/2$$입니다. $$z$$가 아주 큰 음수이면, 분모의 $$e^{-z}$$가 아주 커져서 $$\sigma(z)$$가 0에 수렴합니다. 반대로, $$z$$가 아주 큰 양수이면 $$e^{-z}$$가 0에 가까지므로, $$\sigma(z)$$가 1에 수렴합니다.

시그모이드 함수는 연속 미분 가능한데, 이 도함수는 $$\sigma^\prime(z) = \sigma(z) (1-\sigma(z))$$로 계산이 편리합니다. 신경망을 훈련시키려면 미적분을 사용해야 하기 때문에 이런 도함수가 중요합니다만 당장은 너무 신경쓰지 마세요.

시그모이드 뉴런은 수십 년간 대부분 신경망의 기본이었습니다만, 최근 몇 년동안 크게 선호도가 줄어 들었습니다. 그 이유는 나중에 자세히 설명하겠지만 간단히 말하면, 이 함수가 [그래디언트 소실 문제](https://en.wikipedia.org/wiki/Vanishing_gradient_problem)를 가지고 있어서 많은 층으로 이루어진 신경망을 훈련시키기 어렵게 만들기 때문입니다. 대신에 다른 종류의 활성화 함수인 ReLU를 사용하는 경우가 많습니다. 이름을 봐서는 감을 잡기 어렵지만, 이 함수는 간단하게 $$R(z) = max(0, z)$$로 정의됩니다.

{% include figure_multi.md path1="/images/figures/relu.png" caption1="ReLU 활성화 함수" %}

다른 말로 하면, ReLU는 모든 양수 값은 그냥 통과시키지만, 음수 값은 0으로 만듭니다. 새로운 활성화 함수가 계속 대두되고 있지만, 요즘 대부분 심층 신경망은 ReLU와 그 [변종](https://en.wikipedia.org/wiki/Rectifier_(neural_networks))들 중 하나를 사용합니다.

어떤 활성화 함수를 사용하던지 간에, 다음의 기본적인 그림으로 뉴런 하나의 행동을 직관적이고 시각적으로 표현할 수 있습니다.

{% include figure_multi.md path1="/images/figures/neuron.png" caption1="인공 뉴런" %}

위 그림은 세 개의 입력과 하나의 $$y$$ 값을 출력하는 뉴런을 보여 줍니다. 이전처럼, 먼저 입력의 가중치 합을 계산하고, 그 다음 이를 활성화 함수 $$\sigma$$에 통과시킵니다.

$$
z = b + w_1 x_1 + w_2 x_2 + w_3 x_3 \\
y = \sigma(z)
$$

아마 활성화 함수의 목적이 무엇인지, 왜 이전 장의 선형 분류기에서는 가중치 합을 그냥 출력했는지 궁금해할 것 같습니다. 가중치 합 $$z$$는 입력에 대한 [_선형성_](https://ko.wikipedia.org/wiki/%EC%84%A0%ED%98%95%EC%84%B1), 즉 각 입력에 대해 단순한 의존성을 가집니다. 이와 반대로, 비선형 활성화 함수는 곡선이나 다른 복잡한 함수를 모델링할 수 있습니다. 다음 절에서 좀 더 자세히 나옵니다.

# 층

뉴런에 대해 설명했으니 이제 신경망을 정의할 수 있습니다. 신경망은 뉴런의 연속된 층으로 구성되어 있습니다. 각 층의 뉴런은 다음 층의 뉴런에 모두 연결되어 있습니다.

{% include figure_multi.md path1="/images/figures/neural-net.png" caption1="2개 층으로 된 신경망" %}

신경망에서 층의 수를 헤아릴 때, 입력이 있는 층만 카운트합니다(즉, 첫 번째 입력층은 제외합니다). 그러므로 위의 그림은 1개의 은닉층(hidden layer)을 가진 2층 신경망입니다. 이 신경망은 3개의 입력 뉴런, 은닉층에 2개의 뉴런, 1개의 출력 뉴런을 가지고 있습니다.

왼쪽의 입력층에서 부터 계산이 시작하어 은닉층으로 결과를 전달하고, 다음엔 은닉층의 출력 값을 마지막 층에 보내어 최종 결과를 얻습니다.

세 개의 입력 뉴런이 은닉층의 뉴런 두 개에 모두 연결되어 있기 때문에 여러 개의 값을 보내는 것 같습니다. 하지만 뉴런마다 하나의 출력 값만 있으며, 출력의 연결마다 값이 복사되는 것 뿐입니다. 얼마나 많은 뉴런이 그 값을 받던지 간에 상관없이 뉴런은 항상 하나의 값만 출력합니다.

# 회귀

신경망이 초기 입력을 여러 층을 거쳐 출력까지 전달하는 과정을 정방향 전파(forward propagation) 또는 정방향 패스(forward pass)라고 부릅니다. 그리고 이런 방식의 신경망을 피드포워드(feedforward) 신경망라고 부릅니다. 이제 곧 보게되겠지만 데이터가 순환되는 구조의 신경망도 있습니다만, 아직 너무 앞서 나가지 않는게 좋겠습니다...

인터렉티브한 데모를 사용해 정방향 패스를 시연해 보죠. 오른쪽 위 모서리의 'Next' 버튼을 누르면 시작됩니다.

{% include demo_insert.html path="/demos/simple_forward_pass/" parent_div="post" %}

# 더 많은 층, 더 풍부한 표현력

왜 은닉층이 유용할까요? 그 이유는 은닉층이 없고 입력과 출력을 바로 연결하면, 각 입력이 다른 입력에 상관없이 독립적으로 출력에 기여하기 때문입니다. 실제로는 입력 값이 매우 상호 의존적이고 서로 결합되어 복잡한 구조로 출력에 영향을 미칩니다. 은닉층의 뉴런이 최종 출력에 영향을 미치는 입력간의 미묘한 상호작용을 잡아낼 수 있습니다. 이를 이해하는 다른 방법은 은닉층이 데이터에 있는 고수준의 "특성"이나 속성을 표현한다고 보는 것입니다. 은닉층의 뉴런마다 입력에 다른 가중치를 반영하여 데이터에서 조금 다른 중간적인 특징을 학습하게 되고, 그러면 출력 뉴런은 원본 입력이 아니라 이들의 함수가 됩니다. 은닉층에 더 많은 층를 추가하면, 네트워크가 최종 출력에 도달하기 전에 원본 입력 데이터의 여러 추상 단계를 학습할 기회를 가집니다. 고수준 특성에 대한 개념은 [다음 장](/ml4a/looking_inside_neural_nets/)에서 은닉층에 대해 자세히 살펴 보면 더 명확해 질 것입니다.

활성화 함수도 입력과 출력 사이의 비선형 관계를 잡아내는 데 일조한다는 걸 기억하세요. 여러 층을 통과하면서 여러 비선형 변형이 연결되면, 신경망의 표현력과 유연성이 매우 크게 증가됩니다. 이에 대한 정의는 복잡하고 이 책의 범위를 넘어서지만, 비선형 활성화 함수(시그모이드나 ReLU)와 충분한 은닉 유닛으로 구성된 2층 신경망이 있다면 어떤 [범용 함수](http://www.sciencedirect.com/science/article/pii/0893608089900208)도 근사할 수 있다는 것이 증명되었습니다. 즉 이론적으로 어떤 임의의 입력과 출력의 연결을 표현할 수 있습니다. 이런 성질이 신경망을 강력하게 만듭니다.

# 분류

분류는 어떻게 할까요? 이전 장에서 출력 0을 임계값으로 하는 간단한 이진 분류를 소개했습니다. 출력이 양수이면 양성이라고 분류하고, 음수이면 음성으로 분류했습니다. 신경망에서도 최종 뉴런에 이런 방식을 적용하는 것이 타당해 보입니다. 출력이 어떤 임계값보다 높은 점수를 출력하면 양성으로 분류하는 것입니다. 예를 들어, 항상 양수를 출력하는 시그모이드 뉴런에 대해서는 0.5를 임계값으로 할 수 있습니다.

하지만 여러개의 클래스가 있을 때 어떻게 할까요? 한 가지 방법은 출력 뉴런의 값에 클래스에 상응하는 간격을 만드는 것입니다. [신경망의 훈련 방법](/ml4a/how_neural_networks_are_trained/)에서 배우게 되겠지만 이런 방법에는 문제가 있습니다. 대신에 클래스마다 출력 뉴런 하나를 배정해서 분류를 수행할 수 있습니다. 정방향 패스를 진행해서 가장 높은 값을 내는 뉴런에 상응하는 클래스가 예측이 됩니다. 예제를 한 번 살펴 보죠.

# 손글씨 숫자 분류

그럼 신경망을 사용해서 손글씨 숫자 이미지를 인식하는 실제 예제를 다루어 보죠. 여기서는 [MNIST 데이터셋](http://yann.lecun.com/exdb/mnist/)을 사용하겠습니다. 이 데이터셋은 28x28 픽셀 크기의 레이블된 손글씨 숫자 이미지 60,000개를 담고 있으며, 머신러닝 연구에서 분류 정확도 벤치마크에 널리 사용됩니다.
	
{% include figure_multi.md path1="/images/figures/fig_mnist_groundtruth.png" caption1="MNIST 손글씨 숫자 샘플" %}

이런 이미지를 분류하기 위한 신경망은 원본 픽셀을 첫 번째 층의 입력으로 전달하고, 숫자 클래스 0에서 9까지마다 하나씩 10개의 출력 클래스를 가집니다. 흑백 이미지이기 때문에 각 픽셀은 0(검정색)에서 255(흰색)까지의 밝기를 가집니다. 모든 MNIST 이미지는 28x28이므로 784 픽셀로 구성됩니다. 이를 다음 그림처럼 하나의 입력 배열로 풀어 낼수 있습니다.

{% include figure_multi.md path1="/images/figures/mnist-input.png" caption1="이미지를 신경망에 입력하는 방법" %}

중요한 것은 이 네트워크는 이전에 봤던 단순한 3x2x1 네트워크보다 더 인상적으로 보이지만, 뉴런의 개수만 많을 뿐 하는 일은 정확히 동일하다는 것입니다. 첫 번째 은닉층의 각 뉴런은 이전처럼 첫 번째 층으로 부터 모든 입력을 받습니다. 10개의 출력 뉴런은 각각 하나의 클래스 레이블에 할당됩니다. 첫 번째는 숫자 `0`, 두 번째는 `1` 등의 식입니다.

신경망이 훈련되고 나면 -- [이어지는 장](/ml4a/how_neural_networks_are_trained/)에서 조금 더 자세하게 이야기할 것이 있지만 --, 새로운 샘플을 같은 네트워크에 통과시키고 출력 값을 확인해서 숫자를 예측할 수 있습니다. 예측된 숫자는 가장 높은 출력 값을 가지는 뉴런이 됩니다. 다음 데모는 이 작업을 보여 줍니다. "next"를 누르면 새로운 샘플의 예측을 만듭니다.

{% include demo_insert.html path="/demos/forward_pass_mnist/" parent_div="post" %}

# 더 읽을 거리

{% include further_reading.md title="Neural Networks and Deep Learning" author="Michael Nielsen" link="http://neuralnetworksanddeeplearning.com/" %} 

{% include further_reading.md title="A 'Brief' History of Neural Nets and Deep Learning" author="Andrey Kurenkov" link="http://www.andreykurenkov.com/writing/a-brief-history-of-neural-nets-and-deep-learning/" %} 

{% include further_reading.md title="Demo: Visualization of artificial neural networks" author="Mostafa Abdelraouf" link="http://experiments.mostafa.io/public/ffbpann/" %} 

{% include further_reading.md title="Video: Neural Networks Demystified" author="Welch Labs" link="https://www.youtube.com/watch?v=bxe2T-V8XRs" %} 

## 다음 장

다음 장 [신경망의 내부](/ml4a/looking_inside_neural_nets/)에서 네트워크의 내부 상태를 좀 더 자세히 분석해 보겠습니다. 어떤 종류의 정보를 감지하는지 직관을 얻고, 기본 신경망의 단점은 무엇인지 알아 보겠습니다. 그리고 이를 통해 이어지는 장에서 배울 합성곱 층같이 더 복잡한 기능에 대한 동기를 얻을 것입니다.