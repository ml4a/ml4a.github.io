---
layout: chapter
title: "How neural networks are trained"
includes: [mathjax]
header_image: "/images/headers/topographic_map.jpg"
header_text: "A <a href=\"http://www.summitpost.org/ruth-creek-topographic-map/771858\">topographic map</a> depicts elevation with contour lines connecting places at equal heights."
---
<!--

Gradient descent isn't the only way to solve neural networks. Notably, BGFS (or LBGFS when memory is limited) is sometimes used, but it operates on a similar principle: iterative, small weight updates convering on a good solution. 

todo/more sections?
 - LBGFS, Adam
 - Batchnorm
 - preprocessing (norm, standard), weight init
 - choice of loss function (categorical cross-entropy)
 - use L or C instead of J

-->

[日本語](/ml4a/jp/how_neural_networks_are_trained/)

본인이 산꼭대기에 있는 등산가이고 밤이 깊었다고 상상해 봅시다. 산 아래 베이스캠프에 도착해야 하는데, 희미한 손전등만 있는 이 어둠 속에서는 몇 피트 정도의 땅밖에 보이지 않습니다. 어떻게 내려갈 수 있을까요? 하나의 방법으로는 어느 쪽으로 가장 땅이 기울고 있는지 확인한 다음 그 방향으로 나아가는 것이 있겠습니다. 이 과정을 여러 번 반복하면 내리막길을 따라 점점 아래로 내려갈 수 있습니다. 때때로 작은 골이나 계곡에 갇히게 될 수도 있는데, 이 경우 움직인 방향으로 조금 더 많이 움직여서 계곡을 벗어날 수도 있습니다. 이러한 주의사항만 있으면 이 전략으로 결국 하산할 수 있게 됩니다.

신경망과 전혀 관계가 없는 것처럼 보일 수 있지만, 이 이야기는 신경망이 훈련되는 방식에 대한 좋은 비유입니다. 사실, 그렇게 하기 위한 기본적인 기술인 [경사 하강법](https://ko.wikipedia.org/wiki/%EA%B2%BD%EC%82%AC_%ED%95%98%EA%B0%95%EB%B2%95)은 우리가 방금 설명한 것과 매우 흡사합니다. 훈련은 신경망의 정확도를 최대화하기 위한 최상의 가중치 세트를 결정하는 것을 말합니다. 앞 장에서는 이 과정을 블랙박스 안에 넣어 둔 채로 이미 훈련된 네트워크가 무엇을 할 수 있는지 살펴보았습니다. 그러나 이 장에서는 경사 하강법이 어떻게 작용하는지에 대한 세부 사항을 설명할 것이며, 그것이 방금 설명한 등반가 비유와 아주 유사하다는 것을 알게 될 것입니다.

신경망은 안에 있는 전자제품이 어떻게 작동하는지 모른 채 손전등을 조작할 수 있는 것처럼 훈련 과정을 정확히 알지 못한 채 사용할 수 있습니다. 대부분의 현대 기계 학습 라이브러리는 훈련 과정을 굉장히 자동화했습니다. 자동화된 라이브러리에 의지할 수 있고, 이 주제가 수학적으로 더 어렵기 때문에, 그것을 따로 두고 신경망의 응용 분야로 달려가고 싶은 유혹을 느낄지도 모릅니다. 그러나 그 과정을 이해하는 것은 신경망이 어떻게 적용되고 재구성될 수 있는지에 대한 귀중한 통찰력을 제공하기 때문에, 용감한 독자는 그 과정을 이해하지 않는 것이 실수라는 것을 알고 있을 것입니다. 게다가, 대형 신경망 훈련 능력은 수년 동안 이해할 수 없었고 최근에서야 실현 가능해져, 현재 가장 활동적이고 흥미로운 연구 분야 중 하나일 뿐만 아니라 인공지능 역사상 가장 위대한 성공 사례 중 하나가 되었습니다.

이 장의 목적은 신경망의 해결 방법에 대한 엄격한 이해는 아닐지라도 직관적인 이해를 제공하는 것입니다. 가능한 한 방정식보다 그림으로 설명될 것이며, 추가적인 판독과 정교함을 위한 외부 링크가 제공될 것입니다. 우리는 경사 하강법, 역전파, 그리고 몇 가지 부분에서 관련된 모든 기술에 대해 이야기 할 것입니다. 하지만 우선, 왜 훈련이 어려운지 이해하는 것부터 시작합시다.

# 왜 훈련이 힘든가

## 초차원 건초 더미 안의 바늘

숨겨진 레이어가 있는 신경망의 가중치는 상호의존성이 아주 높습니다. 그 이유를 알기 위해, 아래 세 계층 네트워크의 첫 번째 계층에서 강조 표시된 연결을 고려해봅시다. 만약 우리가 그 연결에서 가중치를 약간 조정한다면, 그 연결이 직접적으로 전파하는 뉴런뿐만 아니라 다음 두 층의 모든 뉴런에도 영향을 미칠 것이고, 따라서 _모든_ 출력에도 영향을 미칠 것입니다.

{% include figure_multi.md path1="/images/figures/connection_tweak.png" caption1="첫 번째 층에서 한 연결의 가중치를 조절하는 것은 다음 층 한 개의 뉴런에만 영향을 미치지만, 완전히 연결되어 있기 때문에 이후 다음 층의 모든 뉴런은 바뀔 것이다." %}

이러한 이유로, 우리는 한 번에 하나씩 최적화함으로써 최고의 가중치 세트를 얻을 수 없다는 것을 알 수 있습니다. 우리는 가능한 전체 가중치 조합을 동시에 찾아야 합니다. 우리가 이걸 어떻게 할 수 있을까요?

가장 단순하고 가장 쉬운 접근법부터 시작해보죠: 무작위 추측입니다. 네트워크의 모든 가중치를 랜덤 값으로 설정하고 데이터 세트에서 정확성을 평가하는 방법입니다. 결과를 추적하면서 이것을 여러 번 반복하고 우리에게 가장 정확한 결과를 준 일련의 가중치를 저장합니다. 처음에 이것은 합리적인 접근으로 보일 수 있습니다. 결국, 컴퓨터는 매우 빠르기 때문에 아마도 우리는 이 무작위 대입 방법으로 괜찮은 해결책을 얻을 수 있을 것입니다. 수십 개의 뉴런이 있는 네트워크라면, 이 방법이 효과가 있을 겁니다. 우리는 빠르게 수백만 가지 추측을 할 수 있고 그것들로부터 괜찮은 후보자를 얻을 수 있을 것입니다. 하지만 대부분의 실제 응용 프로그램에서는 그보다 훨씬 더 많은 가중치를 가집니다. [이전 장](/ml4a/ko/neural_networks/)의 필기 예제를 살펴 봅시다. 약 12,000개의 가중치 값이 있습니다. 그 많은 것들 중에서 가장 좋은 무게의 조합은 말 그대로 건초 더미 안의 바늘과 같습니다. 이 건초 더미는 12,000개의 차원을 가지고 있다는 것을 제외하면요!

여러분은 12,000차원 건초 더미가 더 친숙한 3차원 건초 더미보다 "4,000배 밖에 안 된다"고 생각할지도 모릅니다. 그래서 최고의 가중치를 찾기 위해서는 4,000배의 시간이 필요하다고 말입니다. 하지만 실제로는 그 비율이 이해할 수 없을 정도로 큽니다. 그리고 다음 섹션에서 그 이유를 알아보겠습니다.

## n차원 공간은 외로운 곳입니다.

우리의 전략이 무차별적인 무작위 검색이라면, 합리적으로 좋은 가중치 세트를 얻기 전에 얼마나 많은 추측을 해야 할지 생각해볼 수 있습니다. 직관적으로, 가능한 추측의 전체 공간(space)을 촘촘히 표본으로 추출하기 위해 충분한 추측을 해야 한다고 예상할 수 있습니다. 사전 지식이 없으면 올바른 가중치가 어디에든 숨겨질 수 있기 때문에 가능한 모든 공간을 표본으로 추출하는 것이 타당합니다.

이것을 설명하기 위해, 두 개의 매우 작은 1층 신경망을 생각해 봅시다. 첫번째는 2개의 신경망을 가지고 있고, 두번째는 3개의 신경망을 가지고 있습니다. 단순하게 보기 위해 당분간 편향(bias)는 무시합시다.

{% include figure_multi.md path1="/images/figures/small_nets.png" caption1="각각 두 개의 가중치 연결과 세 개의 가중치 연결을 가진 두 개의 작은 네트워크(당분간 편향은 무시함)." %}

첫 번째 네트워크에는 두 가지 가중치가 있습니다. 그 중 하나가 잘 맞을 거라고 확신하려면 얼마나 많은 추측을 해야 할까요? 이 질문에 접근하는 한 가지 방법은 가능한 체중 조합의 2차원 공간을 상상하고 모든 조합을 어느 정도 세분화 수준까지 철저히 검색하는 것입니다. 아마 우리는 각 축을 10개의 세그먼트로 나눌 수 있습니다. 그러면 우리의 추측은 이 둘의 모든 조합일 것입니다; 모두 100개입니다. 그렇게 나쁘지는 않습니다. 이러한 밀도의 표본 추출은 대부분의 공간을 상당히 잘 고려할 수 있습니다. 축을 10개가 아닌 100개의 세그먼트로 나누면 100*100=10,000개의 추측을 만들어 공간을 매우 촘촘하게 나눠서 고려해야 합니다. 10,000개의 추측은 여전히 매우 작습니다. 어떤 컴퓨터도 1초 안에 계산할 수 있습니다.

두 번째 네트워크는 어떨까요? 여기 두 개 대신 세 개의 가중치가 있고, 따라서 3차원 공간을 탐색해야 합니다. 이 공간을 2차원 네트워크를 샘플링한 것과 동일한 수준의 세분성으로 샘플링하려면 각 축을 다시 10개의 세그먼트로 나눕니다. 이제 10 * 10 * 10 = 1,000개의 추측이 있습니다. 2차원과 3차원 시나리오는 모두 아래 그림에 설명되어 있습니다.

{% include figure_multi.md path1="/images/figures/sampling.png" caption1="왼쪽: 10% 밀도로 표본 추출된 2차원 정사각형에는 10² = 100개의 점이 필요합니다. 오른쪽: 10% 밀도로 표본 추출된 3차원 정육면체에는 10³ = 1000개의 점이 필요합니다." %}

1,000개의 추측은 식은 죽 먹기라고 말할 수 있습니다. 100개의 세그먼트로 세분화하면 $$100 * 100 * 100 = 1000000$$ 추측이 있을 것입니다. 1,000,000개의 추측은 여전히 문제가 되지 않지만, 슬슬 조금씩 긴장이 됩니다. 이 접근 방식을 보다 현실적인 규모의 네트워크로 확장하면 어떻게 됩니까? 가능한 추측의 수는 우리가 가진 가중치의 수와 관련하여 기하급수적으로 증가한다는 것을 알 수 있습니다. 일반적으로, 한 축당 10개의 세그먼트로 세분화하여 샘플링하려면 $$N$$차원 데이터 세트에 $$10^N$$ 샘플이 필요합니다.

그렇다면 이 방법을 사용하여 [첫 번째 장](/ml4a/ko/neural_network/)의 MNIST 숫자를 분류하기 위해 네트워크를 훈련시키려면 어떻게 될까요? 네트워크가 784개의 입력 뉴런, 즉, 15개의 뉴런이 1개의 숨겨진 레이어에 있고 10개의 뉴런이 출력 레이어에 있었습니다. 따라서 $$784*15 + 15*10 = 11910$$의 가중치가 있습니다. 여기에 25개의 편향을 더하면 11,935개의 매개 변수를 통해 동시에 추측해야 합니다. $$10^{11935}$$의 추측을 해야 한다는 뜻이죠. 거의 12,000개의 0이 있는 1입니다! 상상할 수 없을 정도로 큰 숫자입니다. 쉽게 비교하자면, 전 우주에는 $$10^{80}$$개의 원자가 있습니다. 어떤 슈퍼컴퓨터도 그렇게 많은 계산을 할 수는 없습니다. 사실, 만약 오늘날 세상에 존재하는 모든 컴퓨터들을 지구와 태양에 충돌할 때까지 작동시킨다 하더라도, 여전히 계산 중일 것입니다! 현대의 심층 신경망은 수천, 수억개의 가중치를 가지고 있다는 것을 생각해보세요.

이 원리는 우리가 기계 학습에서 "[차원의 저주](https://en.wikipedia.org/wiki/Curse_of_dimensionality)"라고 부르는 것과 밀접하게 관련되어 있습니다. 검색 공간에 추가되는 각 차원은 학습된 모델의 우수한 일반화를 위해 필요한 샘플 수를 기하급수적으로 증가시킵니다. 차원의 저주는 데이터 세트에 적용되는 경우가 더 많습니다. 간단히 말해 데이터 세트가 더 많은 열 또는 변수로 표현될수록 해당 데이터 세트에서 더 많은 샘플을 고려해야 한다는 것입니다. 우리는 입력보다는 가중치에 대해 생각하고 있지만, 원칙은 같습니다; [고차원 공간은 어마어마하다](https://www.inf.fu-berlin.de/inst/ag-ki/rojas_home/documents/tutorials/dimensionality.pdf)!

분명히 이 문제에는 무작위적인 추측보다 더 우아한 해결책이 필요합니다. 이러한 문제를 해결하기 위한 효율적인 계산 방법에 대한 이해를 높이기 위해, 신경망을 잠시 잊고 간단한 문제로 시작하여 경사 하강법을 발명할 때까지 점진적으로 나아가봅시다.

# Linear regression

[Linear regression](https://en.wikipedia.org/wiki/Linear_regression) refers to the task of determining a "line of best fit" through a set of data points and is a simple predecessor to the more complex nonlinear methods we use to solve neural networks. This section will show you an example of linear regression. Suppose we are given a set of 7 points, those in the chart to the bottom left. To the right of the chart is a scatterplot of our points.

{::nomarkdown}
<div style="text-align:center;">
	<div style="display:inline-block; vertical-align:middle; margin-right:100px;">
		<table width="200" style="border: 1px solid black;">
		  	<tbody>
				<tr>
					<td><script type="math/tex">x</script></td>
					<td><script type="math/tex">y</script></td>
				</tr>
				<tr><td>2.4</td><td>1.7</td></tr>
				<tr><td>2.8</td><td>1.85</td></tr>
				<tr><td>3.2</td><td>1.79</td></tr>
				<tr><td>3.6</td><td>1.95</td></tr>
				<tr><td>4.0</td><td>2.1</td></tr>
				<tr><td>4.2</td><td>2.0</td></tr>
				<tr><td>5.0</td><td>2.7</td></tr>
			</tbody>
		</table>
	</div>
	<div style="display:inline-block; vertical-align:middle;">
		<img src="/images/figures/lin_reg_scatter.png">
	</div>
</div>
{:/nomarkdown}

The goal of linear regression is to find a line which best fits these points. Recall that the general equation for a line is $$ f(x) = m \cdot x + b $$, where $$m$$ is the slope of the line, and $$b$$ is its y-intercept. Thus, solving a linear regression is determining the best values for $$m$$ and $$b$$, such that $$f(x)$$ gets as close to $$y$$ as possible. Let's try out a few random candidates.

{% include figure_multi.md path1="/images/figures/lin_reg_randomtries.png" caption1="Three randomly-chosen line candidates" %}

Pretty clearly, the first two lines don't fit our data very well. The third one appears to fit a little better than the other two. But how can we decide this? Formally, we need some way of expressing how good the fit is, and we can do that by defining a loss function.

## Loss function

The loss function -- sometimes called a cost function -- is a measure of the amount of error our linear regression makes on a dataset. Although many loss functions exist, all of them essentially penalize us on the distance between the predicted y-value from a given $$x$$ and its actual value in our dataset. For example, taking the line from the middle example above, $$ f(x) = -0.11 \cdot x + 2.5 $$, we highlight the error margins between the actual and predicted values with red dashed lines.

{% include figure_multi.md path1="/images/figures/lin_reg_error.png" caption1="" %}

One very common loss function is called mean squared error (MSE). To calculate MSE, we simply take all the error bars, square their lengths, and take their average. 

$$ MSE = \frac{1}{n} \sum_i{(y_i - f(x_i))^2} $$

$$ MSE = \frac{1}{n} \sum_i{(y_i - (mx_i + b))^2} $$

We can go ahead and calculate the MSE for each of the three functions we proposed above. If we do so, we see that the first function achieves a MSE of 0.17, the second one is 0.08, and the third gets down to 0.02. Not surprisingly, the third function has the lowest MSE, confirming our guess that it was the line of best fit. 

We can get some intuition if we calculate the MSE for all $$m$$ and $$b$$ within some neighborhood and compare them. Consider the figure below, which uses two different visualizations of the mean squared error in the range where the slope $$m$$ is between -2 and 4, and the intercept $$b$$ is between -6 and 8.

{% include figure_multi.md path1="/images/figures/lin_reg_mse.png" caption1="Left: A graph plotting mean squared error for $ -2 \le m \le 4 $ and $ -6 \le p \le 8 $ <br/>Right: the same figure, but visualized as a 2-d <a href=\"https://en.wikipedia.org/wiki/Contour_line\">contour plot</a> where the contour lines are logarithmically distributed height cross-sections." %}

Looking at the two graphs above, we can see that our MSE is shaped like an elongated bowl, which appears to flatten out in an oval very roughly centered in the neighborhood around $$ (m,p) \approx (0.5, 1.0) $$. In fact, if we plot the MSE of a linear regression for any dataset, we will get a similar shape. Since we are trying to minimize the MSE, we can see that our goal is to figure out where the lowest point in the bowl lies.

## Adding more dimensions

The above example is quite minimal, having just one independent variable, $$x$$, and thus two parameters, $$m$$ and $$b$$. What happens when there are more variables? In general, if there are $$n$$ variables, a linear function of them can be written out as:

$$f(x) = b + w_1 \cdot x_1 + w_2 \cdot x_2 + ... + w_n \cdot x_n $$

Or in matrix notation, we can summarize it as:

$$
f(x) = b + W^\top X
\;\;\;\;\;\;\;\;where\;\;\;\;\;\;\;\;
W = 
\begin{bmatrix}
w_1\\w_2\\\vdots\\w_n\\\end{bmatrix}
\;\;\;\;and\;\;\;\;
X = 
\begin{bmatrix}
x_1\\x_2\\\vdots\\x_n\\\end{bmatrix}
$$

One trick we can use to simplify this is to think of our bias $b$ as being simply another weight, which is always being multiplied by a "dummy" input value of 1. In other words, we let:

$$
f(x) = W^\top X
\;\;\;\;\;\;\;\;where\;\;\;\;\;\;\;\;
W = 
\begin{bmatrix}
b\\w_1\\w_2\\\vdots\\w_n\\\end{bmatrix}
\;\;\;\;and\;\;\;\;
X = 
\begin{bmatrix}
1\\x_1\\x_2\\\vdots\\x_n\\\end{bmatrix}
$$

This equivalent formulation is convenient both notationally, since now our function is more simply expressed as $f(x) = W^\top X$, and conceptually, since we can now think of the bias as just another weight, and therefore just one more parameter that needs to be optimized.

Adding many more dimensions may seem at first to complicate our problem horribly, but it turns out that the formulation of the problem remains exactly the same in 2, 3, or any number of dimensions. Although it is impossible for us to draw it now, there exists a loss function which appears like a bowl in some number of dimensions -- a hyper-bowl! And as before, our goal is to find the lowest part of that bowl, objectively the smallest value that the loss function can have with respect to some parameter selection and dataset.

So how do we actually calculate where that point at the bottom is exactly? There are numerous ways to do so, with the most common approach being the [ordinary least squares](https://en.wikipedia.org/wiki/Ordinary_least_squares) method, which solves it analytically. When there are only one or two parameters to solve, this can be done by hand, and is commonly taught in an introductory course on statistics or linear algebra. 

{% include further_reading.md title="Linear regression tutorial" author="Ozzie Liu" link="http://ozzieliu.com/2016/02/09/gradient-descent-tutorial/" %} 

{% include further_reading.md title="Implementation of linear regression in python" author="Chris Smith" link="https://crsmithdev.com/blog/ml-linear-regression/" %} 

{% include further_reading.md title="Artificial Neural Networks: Linear Regression (Part 1)" author="Brian Dolhansky" link="http://briandolhansky.com/blog/artificial-neural-networks-linear-regression-part-1" %} 

## The curse of nonlinearity

Alas, ordinary least squares cannot be used to optimize neural networks however, and so solving the above linear regression will be left as an exercise left to the reader. The reason we cannot use linear regression is that neural networks are nonlinear; Recall the essential difference between the linear equations we posed and a neural network is the presence of the activation function (e.g. sigmoid, tanh, ReLU, or others). Thus, whereas the linear equation above is simply $$y = b + W^\top X$$, a 1-layer neural network with a sigmoid activation function would be $$f(x) = \sigma (b + W^\top X) $$. 

This nonlinearity means that the parameters do not act independently of each other in influencing the shape of the loss function. Rather than having a bowl shape, the loss function of a neural network is more complicated. It is bumpy and full of hills and troughs. The property of being "bowl-shaped" is called [convexity](https://en.wikipedia.org/wiki/Convex_function), and it is a highly prized convenience in multi-parameter optimization. A convex loss function ensures we have a global minimum (the bottom of the bowl), and that all roads downhill lead to it.

But by introducing the nonlinearity, we lose this convenience for the sake of giving our neural networks much more "flexibility" in modeling arbitrary functions. The price we pay is that there is no easy way to find the minimum in one step analytically anymore (i.e. by deriving neat equations for them). In this case, we are forced to use a multi-step numerical method to arrive at the solution instead. Although several alternative approaches exist, gradient descent remains the most popular and effective. The next section will go over how it works.

# Gradient Descent

The general problem we've been dealing with -- that of finding parameters to satisfy some objective function -- is not specific to machine learning. Indeed it is a very general problem found in [mathematical optimization](https://en.wikipedia.org/wiki/Mathematical_optimization), known to us for a long time, and encountered in far more scenarios than just neural networks. Today, many problems in multivariable function optimization -- including training neural networks -- generally rely on a very effective algorithm called gradient descent to find a good solution much faster than taking random guesses, and more powerful than linear regression. 

## The gradient descent method

Intuitively, the way gradient descent works is similar to the mountain climber analogy we gave in the beginning of the chapter. First, we start with a random guess at the parameters, and start there. We then figure out which direction the loss function steeps downward the most (with respect to changing the parameters), and step slightly in that direction. To put it another way, we determine the amounts to tweak all of the parameters such that the loss function goes down by the largest amount. We repeat this process over and over until we are satisfied we have found the lowest point.

To figure out which direction the loss steeps downward the most, it is necessary to calculate the [gradient](https://en.wikipedia.org/wiki/Gradient) of the loss function with respect to all of the parameters. A gradient is a multidimensional generalization of a [derivative](https://en.wikipedia.org/wiki/Derivative); it is a vector containing each of the partial derivatives of the function with respect to each variable. In other words, it is a vector which contains the slope of the loss function along every axis. 

Although we've already said that the most convenient way to solve linear regression is via ordinary least squares or some other single-step method, let's quickly turn our attention back to linear regression to see a simple example of using gradient descent to solve a linear regression. 

Recall the mean squared error loss we introduced in the previous section, which we will denote as $J$.

$$ J = \frac{1}{n} \sum_i{(y_i - (mx_i + b))^2} $$

There are two parameters we are trying to optimize: $m$ and $b$. Let's calculate the partial derivative of $J$ with respect to each of them. 

$$ \frac{\partial J}{\partial m} = \frac{2}{n} \sum_i{x_i \cdot (y_i - (mx_i + b))} $$

$$ \frac{\partial J}{\partial b} = \frac{2}{n} \sum_i{(y_i - (mx_i + b))} $$

How far in that direction should we step? This turns out to be an important consideration, and in ordinary gradient descent, this is left as a hyperparameter to decide manually. This hyperparameter -- known as the learning rate -- is generally the most important and sensitive hyperparameter to set and is often denoted as $$\alpha$$. If $$\alpha$$ is set too low, it may take an unacceptably long time to get to the bottom. If $$\alpha$$ is too high, we may overshoot the correct path or even climb upwards. 

Denoting the assignment operation as $:=$, we can write the update steps for the two parameters as follows.

$$ m := m - \alpha \cdot \frac{\partial J}{\partial m} $$

$$ b := b - \alpha \cdot \frac{\partial J}{\partial b} $$

If we take this approach to solving the simple linear regression we posed above, we will get something that looks like this:

{% include figure_multi.md path1="/images/figures/lin_reg_mse_gradientdescent.png" caption1="Example of gradient descent for linear regression with two parameters. We take a random guess at the parameters, and iteratively update our position by taking a small step against the direction of the gradient, until we are at the bottom of the loss function." %}

And if there are more dimensions? If we denote all of our parameters as $w_i$, thus giving us the form
$f(x) = b + W^\top X $, then we can extrapolate the above example to the multidimensional case. This can be written down more succinctly using gradient notation. Recall that the gradient of $J$, which we will denote as $\nabla J$, is the vector containing each of the partial derivatives. Thus we can represent the above update step as:

$$ \nabla J(W) = \Biggl(\frac{\partial J}{\partial w_1}, \frac{\partial J}{\partial w_2}, \cdots, \frac{\partial J}{\partial w_N} \Biggr) $$

$$ W := W - \alpha \nabla J(W) $$

The above formula is the canonical formula for ordinary gradient descent. It is guaranteed to get you the best set of parameters for a linear regression, or indeed for any linear optimization problem. If you understand the significance of this formula, you understand "in a nutshell" how neural networks are trained. In practice however, certain things complicate this process in neural networks and the next section will get into how we deal with them.


# Applying gradient descent to neural nets

## The problem of convexity

In the previous section, we showed how to run gradient descent for a simple linear regression problem, and declared that doing so is guaranteed to find the correct parameters. This is true for optimizing a linear model as we did, but it's not true for neural networks, due to the nonlinearity introduced by their activation functions. Consequently, the loss function of a neural net is not "bowl-shaped", and it is not convex. Instead, its loss function is much more complex, with many hills and valleys and curves and other irregularities. This means there are many "local minima" i.e. parameterizations where the loss is the lowest in its own immediate neighborhood, but not necessarily the absolute minimum (or "global minimum"). This means that if we run gradient descent, we might accidentally get stuck in a local minimum.

{% include figure_multi.md path1="/images/figures/non_convex_function.png" caption1="Example of non-convex loss surface with two parameters. Note that in deep neural networks, we're dealing with millions of parameters, but the basic principle stays the sam. Source: <a href=\"http://videolectures.net/site/normal_dl/tag=983679/deeplearning2015_bengio_theoretical_motivations_01.pdf\">Yoshua Bengio</a>." %}

For theoretical reasons beyond the scope of this book, it turns out that this is not a major problem in deep learning, because when there are enough hidden units alongside some other criteria, most local minima are "good enough," being reasonably close to the absolute minimum. According to [Dauphin et al](https://arxiv.org/abs/1406.2572), a bigger challenge than local minima are [saddle points](https://en.wikipedia.org/wiki/Saddle_point), along which the gradient becomes very close to 0. For an explanation of why this is true, see [this lecture](http://videolectures.net/deeplearning2015_bengio_theoretical_motivations/) by [Yoshua Bengio](http://www.iro.umontreal.ca/~bengioy/yoshua_en/) (beginning at section 28, 1:09:41).

Despite the fact that local minima are not a major problem, we'd still prefer to overcome them to the extent they are any problem at all. One way of doing this is to modify the way gradient descent works, which is what the next section is about.

## Stochastic, batch, and mini-batch gradient descent

Besides for local minima, "vanilla" gradient descent has another major problem: it's too slow. A neural net may have hundreds of millions of parameters; this means a single example from our dataset requires hundreds of millions of operations to evaluate. Subsequently, gradient descent evaluated over all of the points in our dataset -- also known as "batch gradient descent" -- is a very expensive and slow operation. Moreover, because every dataset has inherent redundancy, it can be shown that a large enough subset of points can approximate the full gradient anyway, making batch gradient descent unnecessarily expensive to estimate the gradient.

It turns out that we can combat both this problem _and_ the problem of local minima using a modified version of gradient descent called [stochastic gradient descent (SGD)](https://en.wikipedia.org/wiki/Stochastic_gradient_descent). With SGD, we shuffle our dataset, and then go through each sample individually, calculating the gradient with respect to that single point, and performing a weight update for each. This may seem like a bad idea at first because a single example may be an outlier and not necessarily give a good approximation of the actual gradient. But it turns out that if we do this for each sample of our dataset in some random order, the overall fluctuations of the gradient update path will average out and converge towards a good solution. Moreover, SGD helps us get out of local minima and saddle points by making the updates more "jerky" and erratic, which can be enough to get unstuck if we find ourselves in the bottom of a valley. 

SGD is particularly useful in cases where the loss surface is especially irregular. But in general, the usual approach is to use what is called mini-batch gradient descent (MB-GD), in which the whole dataset is randomly subdivided into $$N$$ equally-sized mini-batches of $$K$$ samples each. $$K$$ may be a small positive number, or it can be in the dozens or hundreds; it depends on the specific architecture and application. Note that if $$K=1$$, then you have SGD, and if $$K$$ is the size of the whole dataset, it is batch gradient descent. Note also that confusingly, sometimes people say "SGD" to refer to both MB-GD and one sample at a time.

With MB-GD, we get the best of both worlds; the gradient is smoother and more stable than SGD, and reasonably similar to the full gradient, but we have a massive speed-up from not having to evaluate every sample in the dataset for each update. MB-GD is also computed very efficiently owing to parallelizable matrix operations. 

{% include figure_multi.md path1="/images/figures/bumpy_gradient_descent.png" caption1="Example of gradient descent for non-convex loss function (such as a neural network), with two parameters $\theta_0$ and $\theta_1$. Source: <a href=\"http://www.holehouse.org/mlclass/01_02_Introduction_regression_analysis_and_gr.html\">Andrew Ng</a>." %}

In practice, MB-GD and SGD work well at efficiently optimizing the loss function of a neural network. However, they have weaknesses as well.

 - The aforementioned problem of saddle points; we can get stuck in a parameterization where the loss function plateaus, and the gradient gets very close to 0.
 - The learning rate remains a hyperparameter which must be set manually, which can be difficult to do. A learning rate which is too low leads to slow convergence, and one which is too high may overshoot the correct path. 

## Momentum

[Momentum](https://distill.pub/2017/momentum/) refers to a family of gradient descent variants where the weight update has inertia. In other words, the weight update is no longer a function of just the gradient at the current time step, but is gradually adjusted from the rate of the previous update. 

Recall that in standard gradient descent, we calculate the gradient $$\nabla J(W)$$ and use the following parameter update formula with learning rate $$\alpha$$. 

$$ W_{t} := W_{t} - \alpha \nabla J(W_{t}) $$

Note that we've appended the $$t$$ subscript to denote the current time step, which was previously omitted. In contrast, the generic formula for gradient descent with momentum is the following:

$$ z_{t} := \beta z_{t-1} + \nabla J(W_{t-1}) $$

$$ W_{t} := W_{t-1} - \alpha z_{t} $$

In the parameter update, we've replaced the gradient $$\nabla J(W_{t})$$ with a more complex function $$z_{t}$$ that takes into account the gradient in past time steps. The higher $$\beta$$ is set, the more momentum our parameter update is. If we set $$\beta = 0$$, then the formula reverts to ordinary gradient descent. $$\alpha$$ controls the overall learning rate of the process, as before.

You can think of the update path as being like a ball rolling downhill. Even if it gets to a region where the gradient changes significantly, it will continue going in roughly the same direction under its own momentum, only changing gradually along the path of the gradient. Momentum helps us escape saddle points and local minima by rolling out from them via speed built up from previous updates. It also helps counteract against the common problem of zig-zagging found along locally irregular loss surfaces where the gradient steeps strongly along some directions and not others.

One alternative to the standard momentum formula is Nesterov accelerated gradient descent, given below:

$$ z_{t} := \beta z_{t-1} + \nabla J(W_{t-1} - \beta z_{t-1} ) $$

The only change is, rather than evaluating the gradient where we currently are ($$W_{t-1}$$), we instead evaluate it at approximately where we will be at the next time step ($$W_{t-1} - \beta z_{t-1}$$), given the buildup of momentum carrying us in that direction. Calculating the gradient at that point instead of where we are currently lets us anticipate the loss surface ahead better and tune the momentum term accordingly. An illustration is given below:

{% include figure_multi.md path1="/images/figures/nesterov_acceleration.jpg" caption1="Nesterov momentum \"looks ahead\" to the approximate position we will be in the next update to calculate the gradient term in the update. Source: <a href=\"https://cs231n.github.io/neural-networks-3/\">Stanford CS231n</a>." %}

Momentum methods work pretty well, but like MB-GD and SGD use a single formula for the entire gradient, despite any internal asymmetries among parameters. In contrast, methods which adapt to each element in the gradient have some advantages, which will be looked at in the next section. The following article at [distill.pub](https://distill.pub) looks at momentum in much more mathematical depth and nicely illustrates why it works. 

{% include further_reading.md title="Why momentum works" author="Gabriel Goh" link="https://distill.pub/2017/momentum/" %} 

## Adaptive methods

Momentum comes in many flavors, and in general, finding fast, efficient, and accurate strategies for updating the parameters during gradient descent is a core objective of scientific research in the area, and a full discussion of them is out of the scope of this book. This section will instead quickly survey several of the more prominent variations in practical implementation, and refer to other materials online for a more comprehensive review.

One of the bigger annoyances in the training process is setting the learning rate $$\alpha$$. Typically, an initial $$\alpha$$ is set at the beginning, and is left to decay gradually over some number of time steps, letting it converge more precisely to a good solution. $$\alpha$$ is the same for each individual parameter.

This is unsatisfactory because it assumes that the learning rate must follow a set schedule which is identical for each individual parameter, irrespective of the particular characteristics of the loss surface at a given time step. Additionally, it's unclear how to set $$\alpha$$ and its decay rate in the first place. Momentum and Nesterov momentum help to reduce this burden by giving the update rate some dependence on local observations rather than the "one-size-fits-all" approach of vanilla gradient descent. Still, the choice of $$\alpha$$ and the inflexibility across parameters is seen as a problem.

A number of methods address this shortcoming by adapting the learning rate to each parameter individually, based on the assumption that there is a lot of variance of the loss across all the parameters. The simplest per-parameter update method is [AdaGrad](http://jmlr.org/papers/v12/duchi11a.html) (standing for "Adaptive subGradient"). With AdaGrad, each parameter is updated individually according to its own gradient, but with a new coefficient which attempts to equalize the learning rate between parameters which tend towards large gradients and those that tend to small ones. AdaGrad is defined in the following formula (Note: for the sake of avoiding confusion, note the subscript $$i$$ refers to index of the weight, rather than the time step as before
).

$$ w_{i} := w_{i} - \frac{\alpha}{\sqrt{G_{i}+\epsilon}} \frac{\partial J}{\partial w_{i}} $$

$$\sqrt{G_{i}+\epsilon}$$ represents the sum of the squares of the gradient for that paramter for each step since training began (the $$\epsilon$$ term is just some very small number, e.g. $$10^{-8}$$, to avoid division-by-zero). By dividing $$\alpha$$ for each parameter according to that quantity, we effectively slow down the learning rate for those parameters which have enjoyed large gradients up to that point, and conversely, speed up learning for parameters with minor or sparse gradients.

AdaGrad mostly eliminates the need to treat the initial learning rate $$\alpha$$ as a hyperparameter, but it has its own challenges as well. The typical problem with AdaGrad is that learning may stop prematurely as $$G_{i}$$ accumulates for each parameter over time and reduces the magnitude of the updates. A variant of AdaGrad, [AdaDelta](https://arxiv.org/abs/1212.5701), addresses this by effectivly restricting the window of the gradient accumulation term to the most recent updates. Another adaptive method which is very similar to AdaDelta is [RMSprop](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf). RMSprop -- proposed by [Geoffrey Hinton](http://www.cs.toronto.edu/~hinton/) during his Coursera class but otherwise unpublished -- similarly shortsights the update by summing the squares of the previous updates, but does so in a simpler way by using a standard [easing](http://easings.net) formula with a decay rate (which ends up being a hyperparameter). Thus, for both AdaDelta and RMSprop the update is not just adaptive with respect to parameters, but it's adaptive with respect to time as well, instead of having the learning rate decay monotonically until stopping.

## Adam and comparison of update methods

The last method worth mentioning in this chapter, and one of the most recent to be proposed, is [Adam](http://arxiv.org/abs/1412.6980), whose name is derived from adaptive moment estimation. Adam gives us the best of both worlds between adaptive methods and momentum-based methods. Like AdaDelta and RMSprop, Adam adapts the learning rate for each parameter according to a sliding window of past gradients, but it has a momentum component to smooth the path over time steps.

Still more methods exist, and a full discussion of them is out of the scope of this chapter. A more complete discussion of them, including derivations and practical tips, can be found in [this blog post by Sebastian Ruder](http://ruder.io/optimizing-gradient-descent/index.html).

This nice visualization, courtesy of [Alec Radford](https://twitter.com/alecrad), shows the characteristic behavior among the different gradient update methods discussed so far. Notice that momentum-based methods, Momentum and Nesterov accelerated gradient descent (NAG), tend to overshoot the optimal path by "rolling downhill" too fast, whereas standard SGD moves in the right path, but too slowly. Adaptive methods -- AdaGrad, AdaDelta, and RMSProp (and we could add Adam to it as well) -- tend to have the per-parameter flexibility to avoid both of those trappings.

{% include figure_multi.md path1="/images/figures/opt2a.gif" caption1="Contour plot of gradient update methods converging on good parameters. Figure by <a href=\"https://www.twitter.com/alecrad\">Alec Radford</a>" path2="/images/figures/opt1a.gif" caption2="Comparison of gradient update methods escaping from a saddle point. Notice that SGD gets stuck. Figure by <a href=\"https://www.twitter.com/alecrad\">Alec Radford</a>" %}

So which optimization method works best? There's no simple answer to this, and the answer largely depends on the characteristics of your data and other training constraints and considerations. Nevertheless, Adam has emerged as a promising method to at least start with. When data is sparse or unevenly distributed, the purely adaptive methods tend to work best. A full discussion of when to use each method is beyond the scope of this chapter, and is best found in the academic papers on optimizers, or in practical summaries such as [this one by Yoshua Bengio](https://arxiv.org/pdf/1206.5533v2.pdf).

For further reading on gradient descent optimization, see the following:

{% include further_reading.md title="An overview of gradient descent optimization algorithms" author="Sebastian Ruder" link="http://ruder.io/optimizing-gradient-descent/index.html" %} 

{% include further_reading.md title="Optimizing convolutional networks (CS231n)" author="Andrej Karpathy" link="https://cs231n.github.io/neural-networks-3/" %} 


# Hyperparameters and evaluation

Now that we understand the notion of optimizing the parameters of a network, we are ready to summarize the full procedure. The naive way to train our final model would be to run the gradient descent procedure over our full data. But we run into a problem if we do this: how do we evaluate the accuracy of our model? Since we've used up all our labeled data for training, the only way to evaluate it is to run the model on our training set again, and measure the difference between the output and the "ground truth" (the given labels). To understand why this is a bad practice, it is necessary to understand the phenomenon of overfitting.

## Overfitting

[Overfitting](https://en.wikipedia.org/wiki/Overfitting) describes the situation in which your model is over-optimized to accurately predict the training set, at the expense of generalizing to unknown data (which is the objective of learning in the first place). This can happen because the model greatly twists itself to perfectly conform to the training set, even capturing its underlying noise. 

One way we can think of overfitting is that our algorithm is sort of \"cheating.\" It is trying to convince you it has an artificially high score by orienting itself in such a way as to get minimal error on the known data. It would be as though you are trying to learn how fashion works but all you've seen is pictures of people at disco nightclubs in the 70s, so you assume all fashion everywhere consists of nothing but bell bottoms, denim jackets, and platform shoes. Perhaps you even have a close friend or family member whom this describes.

An example of this can be seen in the below graph. We are given 11 data points in black, and two functions are trained to fit it. One is a straight line, which roughly captures the data. The other is a very curvy line, which perfectly captures the data with no error. The latter may at first seem like the better fit because it has less (indeed, zero) error on the training data. But it probably does not really capture the underlying distribution very well and would have poor performance on unknown points.

{% include figure_multi.md path1="/images/figures/overfitting.png" caption1="An example of overfitting. The straight line is simple and roughly captures the data points with some error. The curvy line has 0 error but is very complex and likely does not generalize well. Source: <a href=\"https://en.wikipedia.org/wiki/Overfitting\">Wikipedia</a>." %}

How can we avoid overfitting? The simplest solution is to split our dataset into a training set and a test set. The training set is used for the optimization procedure we described above, but we evaluate the accuracy of our model by forwarding the test set to the trained model and measuring its accuracy. Because the test set is held out from training, this prevents the model from "cheating," i.e. memorizing the samples it will be quizzed on later. During training, we can monitor the accuracy of the model on the training set and test set. The longer we train, the more likely our training accuracy is to go higher and higher, but at some point, it is likely the test set will stop improving. This is a cue to stop training at that point. We should generally expect that training accuracy is higher than test accuracy, but if it is much higher, that is a clue that we have overfit.

## Cross-validation and hyperparameter selection

The procedure above is a good start to combat overfitting, but it turns out to be not enough. There remain a number of crucial decisions to make before optimization begins. What model architecture should we use? How many layers and hidden units should there be? How should we set the learning rate and other hyperparameters? We could simply try different settings, and pick the one that has the best performance on the test set. But the problem is we risk setting the hyperparameters to be those values which optimize only _that particular_ test set, rather than an arbitrary or unknown one. This would again mean that we are overfitting.

The solution to this is to partition the data into three sets rather than two: a training set, a validation set, and a test set. Typically you will see splits where the training set accounts for 70 or 80% of the full data, and the test and validation are equally split among the rest. Now, you train on the training set, and evaluate on the validation set in order to find the optimal hyperparameters and know when to stop training (typically when validation set accuracy stops improving). Sometimes, cross-fold validation is preferred; in this type of setup, the training and validation set is split into some number (e.g. 10) equally-sized partitions, and each partition takes turns being the validation set. Other times, one validation set is used persistently. After validation, the final evaluation is carried out on the test data, which has been held out the whole time leading up to the end.

Recently, a number of researchers have even begun devising ways of learning architectures and hyperparameters within the training process itself. Researchers at [Google Brain](https://research.google.com/teams/brain/) call this [AutoML](https://research.googleblog.com/2017/05/using-machine-learning-to-explore.html). Such methods hold great potential in automating those tedious components of machine learning which still require human intervention, and perhaps point to a future when someone will need only to define a problem and provide a dataset in order to do machine learning.

## Regularization

[Regularization](https://en.wikipedia.org/wiki/Regularization_(mathematics)) refers to imposing constraints on our neural network in order to prevent overfitting or otherwise discourage undesirable properties. One way overfitting occurs is when the magnitude of the weights grows too large; it is this property that allows the shape of the network output function to curve so wildly as to capture the underlying noise of a training set, as we saw in the above example.  

One way to regularize is to modify our objective function by adding an additional term which penalizes large weights. Denoting our neural network as $$f$$, recall that the loss function we are optimizing is the mean squared error:

$$ J = \frac{1}{n} \sum_i{(y_i - f(x_i))^2} $$

We can penalize large weights by appending our loss function with the L2-regularization term, denoted here as $R(f)$:

$$ R(f) = \frac{1}{2} \lambda \sum{w^2} $$

This term is simply the sum of the squares of all of the weights, multiplied by a new hyperparameter $\lambda$ which controls the overall magnitude (and therefore influence) of the regularization term. The $\frac{1}{2}$ multiplier is simply used for convenience when taking its derivative. Adding it to our original loss function, we now have:

$$ J = \frac{1}{n} \sum_i{(y_i - f(x_i))^2} + R(f) $$

The effect of this regularization term is is that we help gradient descent find a parameterization which does not accumulate large weights and have such wild swings as we saw above.

Other regularization terms exist, including [L1-distance](https://en.wikipedia.org/wiki/Taxicab_geometry) or the "Manhattan distance." Each of these have slightly different properties but have approximately the same effect.

## Dropout

Dropout is a clever technique for regularization, which was only introduced by [Nitish Srivastava et al](http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf) in 2014. During training, when dropout is applied to a layer, some percentage of its neurons (a hyperparameter, with common values being between 20 and 50%) are randomly deactivated or "dropped out," along with their connections. Which neurons are dropped out are constantly shuffled randomly during training. The effect of this is to reduce the network's tendency to come to over-depend on some neurons, since it can't rely on them being available all the time. This forces the network to learn a more balanced representation, and helps combat overfitting. It is depicted below, from its original publication.

{% include figure_multi.md path1="/images/figures/dropout.png" caption1="During training, dropout randomly deactivates some neurons as a method for combatting overfitting. Source: <a href=\"http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf\">Srivastava et al</a>." %}

Another exotic method for regularization is [adding a bit of noise to the inputs](https://www.microsoft.com/en-us/research/publication/training-with-noise-is-equivalent-to-tikhonov-regularization/). Still many others have been proposed with varying levels of success, but will not be covered in-depth here.

# Backpropagation

At this point, we've introduced the gradient descent algorithm for parameterizing neural networks, along with a number of flavored alternatives including adaptive and momentum-based methods. Regardless of the exact variant chosen, all of them need to compute the gradient of the loss function with respect to the weights and biases of the network. This is no easy task. To see why, let's think about how we might go about doing this. 

Recall that our weight update formula in standard gradient descent is given by the following:

$$ W_{t} := W_{t} - \alpha \nabla J(W_{t}) $$

$\nabla J(W_{t})$ is the gradient of the loss, and must be computed in some form across all of the gradient descent varieties we surveyed. Recall that the gradient is a vector which contains each of the individual partial derivatives of the cost function with respect to each parameter, and is given by the following ($t$ is omitted for brevity). 

$$ \nabla J(W) = \Biggl(\frac{\partial J}{\partial w_1}, \frac{\partial J}{\partial w_2}, \cdots, \frac{\partial J}{\partial w_N} \Biggr) $$

How can we calculate each $\frac{\partial J}{\partial w_i}$? The most obvious way to do this would be to compute it with the equation for a derivative from ordinary calculus:

$$ \frac{\partial J}{\partial w_i} \approx \frac{J(W + \epsilon e_i) - J(W)}{\epsilon} $$

Where $e_i$ is a one-hot vector (all zeros except 1 at index $i$) and $\epsilon$ is a some very small number. Technically this will work, but it presents us with a major problem: speed. To get a single element of the gradient, it's necessary to calculate the the loss function at both $W + \epsilon e_i$ and $W$. For $W$ it's only necessary to do this once, but we need $J(W + \epsilon e_i)$ for every single weight $w_i$. Typical deep neural nets have millions or even hundreds of millions of weights. This would entail doing millions of forward passes, each of which has millions of operations, just to do a single weight update. This is totally impractical for training neural nets.

So how do we do it? In fact, until the development of [backpropagation](https://en.wikipedia.org/wiki/Backpropagation), this was a major impediment to training neural networks. The question of who invented backpropagation ("backprop" for short) is a [contentious issue](https://plus.google.com/100849856540000067209/posts/9BDtGwCDL7D), and it seems that a number of people have re-invented it at different times throughout history, or stumbled upon similar concepts applied to different problems. Although largely associated with neural networks, backprop can be used on any problem that involves calculating a gradient on a continuously differentiable multivariate function, and as such, its development was somewhat parallel to the development of neural networks in general. In 2014, [Jürgen Schmidhuber](http://www.idsia.ch/~juergen) compiled a [review of the relevant work that went into developing backprop](http://people.idsia.ch/~juergen/who-invented-backpropagation.html). 

Backpropagation was first applied to the task of optimizing neural networks by gradient descent in a [landmark paper in 1986](https://www.iro.umontreal.ca/~vincentp/ift3395/lectures/backprop_old.pdf) by [David Rumelhart](https://en.wikipedia.org/wiki/David_Rumelhart), [Geoffrey Hinton](http://www.cs.toronto.edu/~hinton/), and [Ronald J. Williams](http://www.ccs.neu.edu/home/rjw/). Subsequent work was done in the 80s and 90s by [Yann LeCun](http://yann.lecun.com/ex/research/index.html), who first applied it to convolutional networks. The success of neural networks was largely enabled by their efforts along with their teams.

A full explanation of how backpropagation works is beyond the scope of this book. Instead, this paragraph will offer a basic high-level view of what backprop gives us, and defer a more technical explanation of it to a number of sources for further reading. The basic idea is that backprop makes it possible to compute all the elements of the gradient in a single forward and backward pass through the network, rather than having to do one forward pass for _every single_ parameter, as we'd have to using the naive approach. This is enabled by utilizing [the chain rule](https://en.wikipedia.org/wiki/Chain_rule) in calculus, which lets us decompose a derivative as a product of its individual functional parts. By keeping track of the differences in a forward pass along every connection and storing them, we can calculate the gradient by taking the loss term found at the end of the forward pass, and "propagating the error backwards" through each of the layers. This makes a backward pass take roughly the same amount of work as a forwards pass. This dramatically speeds up training and makes doing gradient descent on deep neural networks a feasible problem.

For more in-depth technical explanations of how backprop is derived, see the following links for further reading.

{% include further_reading.md title="How the backpropagation algorithm works" author="Michael Nielsen" link="http://neuralnetworksanddeeplearning.com/chap2.html" description="A free online book which introduces neural networks and deep learning from scratch" %} 

{% include further_reading.md title="Hacker's guide to Neural Networks" author="Andrej Karpathy" link="http://karpathy.github.io/neuralnets/" %} 

{% include further_reading.md title="Deep Learning Basics: Neural Networks and Stochastic Gradient Descent" author="Alex Minnaar" link="http://alexminnaar.com/deep-learning-basics-neural-networks-backpropagation-and-stochastic-gradient-descent.html" %} 
 
{% include further_reading.md title="[Video] Back Propagation Derivation for Feed Forward Artificial Neural Networks " author="Sully Chen" link="https://www.youtube.com/watch?v=gl3lfL-g5mA" %} 

{% include further_reading.md title="[Video] Neural network tutorial: the back-propagation algorithm (2 parts)" author="Ryan Harris" link="https://www.youtube.com/watch?v=aVId8KMsdUU" %} 

{% include further_reading.md title="Calculus on Computational Graphs: Backpropagation" author="Chris Olah" link="colah.github.io/posts/2015-08-Backprop/" %} 

{% include further_reading.md title="A Step by Step Backpropagation Example" author="Matt Mazur" link="https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/" %} 

# Descending the mountain

If you've made it this far into the article, then by now the analogy of the mountain climber put forth in the beginning of this chapter should be beginning to make sense to you. If that's the case, congratulations: you appreciate the art and science of how neural networks are trained to a sufficient enough degree that actual scientific research into the topic should seem much more approachable. As the years have gone on, many scientists have proposed various and exotic extensions to backpropagation. Others, including Geoffrey Hinton himself, have suggested that machine learning [must move on from backpropagation and start over](http://www.i-programmer.info/news/105-artificial-intelligence/11135--geoffrey-hinton-says-ai-needs-to-start-over.html). But as of the writing of this book, gradient descent via backpropagation continues to be the dominant paradigm for training neural networks and most other machine learning models, and looks to be set to continue on that path for the foreseeable future. 

In the next few chapters of the book, we are going to start to look at more advanced topics. We will introduce [Convolutional neural networks](/ml4a/convnets/) in the next chapter, as well as their numerous applications, especially toward art and other creative purposes that are at the heart of this book.

