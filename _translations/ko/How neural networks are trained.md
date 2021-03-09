
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

그렇다면 이 방법을 사용하여 [첫 번째 장](/ml4a/ko/neural_network/)의 MNIST 숫자를 분류하기 위한 네트워크를 훈련시키려면 어떻게 될까요? 네트워크가 784개의 입력 뉴런, 즉, 15개의 뉴런이 1개의 숨겨진 층에 있고 10개의 뉴런이 출력층에 있었습니다. 따라서 $$784*15 + 15*10 = 11910$$의 가중치가 있습니다. 여기에 25개의 편향을 더하면 11,935개의 매개 변수를 통해 동시에 추측해야 합니다. $$10^{11935}$$의 추측을 해야 한다는 뜻이죠. 거의 12,000개의 0이 있는 1입니다! 상상할 수 없을 정도로 큰 숫자입니다. 쉽게 비교하자면, 전 우주에는 $$10^{80}$$개의 원자가 있습니다. 어떤 슈퍼컴퓨터도 그렇게 많은 계산을 할 수는 없습니다. 사실, 만약 오늘날 세상에 존재하는 모든 컴퓨터들을 지구와 태양에 충돌할 때까지 작동시킨다 하더라도, 여전히 계산 중일 것입니다! 현대의 심층 신경망은 수천, 수억개의 가중치를 가지고 있다는 것을 생각해보세요.

이 원리는 우리가 기계 학습에서 "[차원의 저주](https://en.wikipedia.org/wiki/Curse_of_dimensionality)"라고 부르는 것과 밀접하게 관련되어 있습니다. 검색 공간에 추가되는 각 차원은 학습된 모델의 우수한 일반화를 위해 필요한 샘플 수를 기하급수적으로 증가시킵니다. 차원의 저주는 데이터 세트에 적용되는 경우가 더 많습니다. 간단히 말해 데이터 세트가 더 많은 열 또는 변수로 표현될수록 해당 데이터 세트에서 더 많은 샘플을 고려해야 한다는 것입니다. 우리는 입력보다는 가중치에 대해 생각하고 있지만, 원칙은 같습니다; [고차원 공간은 어마어마하다](https://www.inf.fu-berlin.de/inst/ag-ki/rojas_home/documents/tutorials/dimensionality.pdf)!

분명히 이 문제에는 무작위적인 추측보다 더 우아한 해결책이 필요합니다. 이러한 문제를 해결하기 위한 효율적인 계산 방법에 대한 이해를 높이기 위해, 신경망을 잠시 잊고 간단한 문제로 시작하여 경사 하강법에 도달할 때까지 점진적으로 나아가봅시다.

# 선형 회귀

[선형 회귀](https://ko.wikipedia.org/wiki/선형_회귀)는 일련의 데이터 포인트를 통해 "가장 적합한 선"을 결정하는 작업을 말하며 신경망 해결에 사용하는 보다 복잡한 비선형 방법에 선행하는 단순한 방법입니다. 이 절에서는 선형 회귀 분석의 예를 보여 줍니다. 아래 그림의 왼쪽 표와 같이 7개의 점으로 이루어진 집합이 있다고 가정해봅시다. 오른쪽에는 이 점들의 산점도가 있습니다.

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

선형 회귀 분석의 목적은 이러한 점에 가장 적합한 1차 함수를 찾는 것입니다. 1차 함수의 일반 방정식은 $$ f(x) = m \cdot x + b $$이며, 여기서 $$m$$는 기울기이고 $$b$$는 y절편입니다. 따라서, 선형 회귀를 해결하는 것은 $$f(x)$$가 $$y$$와 최대한 비슷하도록 최상의 $$m$$과 $$b$$의 값을 결정하는 것입니다. 무작위로 몇 가지 후보들을 시험해 봅시다.

{% include figure_multi.md path1="/images/figures/lin_reg_randomtries.png" caption1="무작위로 선택된 1차 함수 후보 3가지" %}

분명히 처음 두 1차 함수들은 데이터에 잘 맞지 않습니다. 세 번째 것은 다른 두 개보다 조금 더 잘 맞는 것 같습니다. 하지만 어떻게 판단할 수 있을까요? 우리는 얼마나 적합한지를 표현할 수 있는 공식적인 방법이 필요하고, 바로 손실 함수를 정의해서 이를 표현할 수 있습니다.

## 손실 함수
손실 함수(또는 비용 함수라고도 함)는 선형 회귀 분석 중 데이터 세트에서 발생하는 오류의 양을 측정하는 것입니다. 많은 손실 함수가 존재하지만, 모든 함수는 기본적으로 주어진 $$x$$에서 예측된 y 값과 실제 값 사이의 거리에 대해 불이익을 줍니다. 예를 들어, 위의 중간 예에서 나온 1차 함수를 보면 $$f(x) = -0.11 \cdot x + 2.5$$의 실제 값과 빨간색 점선으로 예측된 값 사이의 오차 한계를 주목합니다.

{% include figure_multi.md path1="/images/figures/lin_reg_error.png" caption1="" %}

매우 일반적인 손실 함수 중 평균 제곱 오차(MSE)가 있습니다. MSE를 계산하려면 모든 오류 막대를 선택하고 길이를 제곱한 다음 평균을 구하면 됩니다.

$$ MSE = \frac{1}{n} \sum_i{(y_i - f(x_i))^2} $$

$$ MSE = \frac{1}{n} \sum_i{(y_i - (mx_i + b))^2} $$

우리는 앞서 제안한 세 1차 함수 각각에 대한 MSE를 계산할 수 있습니다. 이렇게 하면 첫 번째 함수는 0.17의 MSE 값을 가지고 있고, 두 번째 함수는 0.08이며, 세 번째 함수는 0.02로 내려갑니다. 놀랄 것도 없이, 세 번째 함수는 MSE가 가장 낮아서 그것이 가장 적합하다는 우리의 추측을 확인시켜줍니다.

어떤 이웃한 1차 함수들에서 모든 $$m$$와 $$b$$에 대한 MSE를 계산하고 비교한다면 직관을 얻을 수 있습니다. 아래 그림을 생각해 보십시오. 이 그림은 기울기 $$m$$가 -2와 4 사이이고 절편 $$b$$가 -6과 8 사이인 범위에서 평균 제곱 오차를 두 가지 방법으로 시각화했습니다.

{% include figure_multi.md path1="/images/figures/lin_reg_mse.png" caption1="왼쪽: $ -2 \lem \le 4$ 및 $ -6 \le p \le 8 $에 대한 평균 제곱 오차를 표시한 그래프 <br/> 오른쪽: 동일 값을 표시한, 등고선도가 로그적으로 분포된 높이 횡단면이 있는 2차원 <ref=\"https://en.wikipedia.org/wiki/Contour_line\">등치선</a>" %}

위의 두 그래프를 보면, 우리의 MSE는 길쭉한 사발처럼 생겼다는 것을 알 수 있습니다. 이 사발은 이웃에 있는 대략 $$ (m,p) \approx (0.5, 1.0)$$의 타원형으로 평평하게 보입니다. 실제로 데이터 세트에 대한 선형 회귀 분석의 MSE를 그려보면 유사한 모양이 나타납니다. MSE를 최소화하기 위해 노력하고 있기 때문에 그릇에 담긴 가장 낮은 지점이 어디에 있는지 파악하는 것이 우리의 목표임을 알 수 있습니다.

## 더 많은 차원을 더하기

위의 예는 매우 작은 것으로, 하나의 독립 변수인 $$x$$와 $$m$$와 $$b$$의 두 가지 매개 변수를 가지고 있습니다. 변수가 더 많으면 어떻게 될까요? 일반적으로 $$n$$개의 변수가 있는 경우, 변수의 1차 선형 함수는 다음과 같이 작성할 수 있습니다.

$$f(x) = b + w_1 \cdot x_1 + w_2 \cdot x_2 + ... + w_n \cdot x_n $$

행렬로 표시하자면, 이렇게 요약할 수 있습니다:

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

이를 단순화하기 위해 사용할 수 있는 한 가지 방법은 바이어스 $b$를 단순히 다른 가중치로 생각하는 것입니다. 이 가중치는 항상 "더미" 입력 값 1에 곱해 표현됩니다. 즉, 다음과 같이 표현할 수 있습니다.

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

이 등가 공식은 함수를 더 단순하게 $f(x) = W^\top X$로 표현할 수 있고, 편향을 가중치 중 하나로 생각하면, 매개 변수 하나만 최적화하면 되므로 아주 합리적인 방법입니다.

더 많은 차원을 추가하는 것은 처음에는 우리의 문제를 끔찍히 복잡하게 만드는 것처럼 보일였지만, 알고 보니 문제의 공식은 2, 3 또는 어떤 수의 차원으로도 정확히 동일하게 유지됩니다. 비록 지금 그것을 그리는 것은 불가능하지만, 어떤 차원에서는 그릇처럼 보이는 손실 함수가 존재합니다. -- 초-사발(hyper-bowl)이죠! 그리고 이전과 마찬가지로, 우리의 목표는 그 그릇의 가장 낮은 부분, 객관적으로 손실 함수가 특정 매개 변수와 데이터 세트에 대해 가질 수 있는 가장 작은 값을 찾는 것입니다.

그러면 어떻게 하면 맨 아래에 있는 그 지점이 정확히 어디에 있는지 계산할 수 있을까요? 가장 일반적인 방법은 분석적으로 해결하는 [최소 제곱법](https://ko.wikipedia.org/wiki/최소제곱법)이 있지만, 다양한 방법을 사용할 수 있습니다. 풀어야 할 매개변수가 하나 또는 두 개뿐일 때, 이것은 손으로 할 수 있고, 일반적으로 통계학이나 선형 대수학 입문 과정에서 배울 수 있습니다.

{% include further_reading.md title="Linear regression tutorial" author="Ozzie Liu" link="http://ozzieliu.com/2016/02/09/gradient-descent-tutorial/" %} 

{% include further_reading.md title="Implementation of linear regression in python" author="Chris Smith" link="https://crsmithdev.com/blog/ml-linear-regression/" %} 

{% include further_reading.md title="Artificial Neural Networks: Linear Regression (Part 1)" author="Brian Dolhansky" link="http://briandolhansky.com/blog/artificial-neural-networks-linear-regression-part-1" %} 

## 비선형성의 저주

그러나, 일반적인 최소 제곱은 신경망 최적화를 위해 사용될 수 없으므로, 위의 선형 회귀 분석을 해결하는 것은 여러분께 연습으로 남겨놓겠습니다. 선형 회귀를 사용할 수 없는 이유는 신경망은 비선형적이기 때문입니다. 우리가 제시한 선형 방정식과 신경망 사이의 본질적인 차이는 활성화 함수(예: 시그모이드, tanh, ReLU 또는 기타)의 존재입니다. 따라서, 위의 선형 방정식은 단순히 $y = b + W^\top X$인 반면, 시그모이드 활성화 함수를 갖는 단층 신경망은 $f(x) = \two(b + W^\top X)$가 될 것입니다.

이 비선형성은 매개 변수가 손실 함수의 모양에 영향을 미치는 데 서로 독립적으로 작용하지 않음을 의미합니다. 신경망의 손실 함수는 그릇 모양보다 더 복잡합니다. 울퉁불퉁하고 언덕과 수조가 가득합니다. "그릇 모양" 특징은 [볼록 함수](https://ko.wikipedia.org/wiki/볼록_함수)라고 하며, 다중 파라미터를 최적화할 때 매우 편리합니다. 볼록 손실 함수는 글로벌 최소값(그릇의 바닥)을 보장하며 내리막길의 모든 도로가 볼록한 상태로 이어지도록 합니다.

그러나 비선형성을 도입함으로써, 우리는 신경망의 함수를 모델링하는 데 훨씬 더 많은 "유연성"을 제공하기 위해 이러한 편리성을 잃게 됩니다. 더 이상 분석적으로 최소값을 찾을 수 있는 쉬운 방법이 없다는 것입니다. 이 경우, 우리는 정답에 도달하기 위해 다단계의 수치적인 방법을 사용해야 합니다. 몇 가지 대안적 접근법이 존재하지만, 경사 하강법은 여전히 가장 대중적이고 효과적입니다. 다음 절에서는 어떻게 작동하는지 살펴보겠습니다.

# 경사 하강

우리가 다루어 온 일반적인 문제, 즉 어떤 객관적인 기능을 만족시키기 위한 매개 변수를 찾는 문제는 기계 학습에만 국한되지 않습니다. 실제로 그것은 오랫동안 알려진 [수학적 최적화](https://ko.wikipedia.org/wiki/수학적_최적화),에서 발견되는 매우 일반적인 문제이며, 단순한 신경망보다 훨씬 더 많은 시나리오에서 발견되었습니다. 오늘날, 신경망 훈련을 포함한 다변수 함수 최적화의 많은 문제들은 일반적으로 무작위 추측보다 훨씬 빠르고 선형 회귀보다 더 강력한 해결책을 찾기 위해 경사 하강법라고 불리는 매우 효과적인 알고리즘에 의존합니다.

## 경사 하강법

직관적으로, 경사 하강법이 작동하는 방식은 우리가 챕터 앞부분에서 제시한 산악인의 비유와 비슷합니다. 먼저, 매개 변수를 랜덤하게 추측하는 것으로 시작합니다. 그런 다음 매개 변수 변경과 관련하여 손실 함수가 가장 아래로 기울어지는 방향을 파악하고 해당 방향으로 약간 이동합니다. 다시 말하면 손실 함수가 가장 큰 폭으로 감소하도록 모든 매개 변수를 조정할 양을 결정합니다. 우리는 우리가 가장 낮은 점을 발견했다고 만족할 때까지 이 과정을 계속해서 반복합니다.

손실 함수가 가장 아래로 기울어지는 방향을 파악하려면 모든 매개 변수에 대해 손실 함수의 [기울기](https://ko.wikipedia.org/wiki/기울기)을 계산해야 합니다. 기울기(gradient)는 [미분](https://ko.wikipedia.org/wiki/미분)의 다차원 일반화입니다. 이는 각 변수에 대한 함수의 편미분을 포함하는 벡터입니다. 즉, 모든 축에 대한 손실 함수의 기울기를 포함한 벡터입니다.

선형 회귀를 해결하는 가장 편리한 방법은 일반 최소 제곱법이나 다른 단일 단계 방법을 사용하는 것이라고 이미 언급했지만, 선형 회귀 분석을 위한 경사 하강법을 사용하는 간단한 예를 보기 위해 선형 회귀 분석으로 다시 돌아가 보겠습니다.

이전 섹션에서 소개한 평균 제곱 오차 손실을 다시 생각해봅시다. 이를 $J$라고 합니다.

$$ J = \frac{1}{n} \sum_i{(y_i - (mx_i + b))^2} $$

최적화하려는 매개 변수는 $m$와 $b$입니다. 각각에 대해 $J$의 편미분을 계산해 보겠습니다.

$$ \frac{\partial J}{\partial m} = \frac{2}{n} \sum_i{x_i \cdot (y_i - (mx_i + b))} $$

$$ \frac{\partial J}{\partial b} = \frac{2}{n} \sum_i{(y_i - (mx_i + b))} $$

그 방향으로 얼마나 더 나아가야 할까요? 이것은 중요한 고려사항으로 밝혀졌습니다. 그리고 보통의 경사 하강법에서는, 이것은 수동으로 결정하는 하이퍼 파라미터(hyperparameter)로 남겨집니다. 학습률로 알려진 이 하이퍼 파라미터는 일반적으로 가장 중요하고 민감한 하이퍼 파라미터이며 종종 $$\alpha$$로 표시됩니다. $$\alpha$$가 너무 낮게 설정되어 있으면 가장 낮은 곳으로 이동하는 데 참을 수 없을 정도로 오랜 시간이 걸릴 수 있습니다. $$\alpha$$가 너무 높으면 올바른 경로를 오버슈팅(overshoot)하거나 심지어는 위로 올라갈 수도 있습니다.

할당 작업을 $:=$로 나타내면 두 매개 변수에 대한 업데이트 단계를 다음과 같이 작성할 수 있습니다.

$$ m := m - \alpha \cdot \frac{\partial J}{\partial m} $$

$$ b := b - \alpha \cdot \frac{\partial J}{\partial b} $$

위에서 설명한 간단한 선형 회귀 분석을 해결하기 위해서 이 방법을 사용하면 다음과 같은 결과를 얻을 수 있습니다.

{% include figure_multi.md path1="/images/figures/lin_reg_mse_gradientdescent.png" caption1="두 개의 매개 변수를 사용한 선형 회귀 분석에 대한 경사 하강 예입니다. 매개 변수를 랜덤하게 추측하고 손실 함수의 맨 아래에 도달할 때까지 기울기 방향으로 조금씩 이동해 반복적으로 위치를 업데이트합니다." %}

그리고 만약 더 많은 차원이 있다면요? 모든 매개 변수를 $w_i$로 표시하면,
$f(x) = b + W^\top X$로 표시할 수 있습니다. 그런 다음 위의 예를 다차원 사례에 대해 추론할 수 있습니다. 기울기 표기법을 사용하여 보다 간결하게 표현할 수 있습니다. $\nabla J$로 표현하는 $J$의 기울기는 각각의 편미분을 포함하는 벡터임을 잊지마세요. 따라서 위의 업데이트 단계를 다음과 같이 나타낼 수 있습니다.

$$ \nabla J(W) = \Biggl(\frac{\partial J}{\partial w_1}, \frac{\partial J}{\partial w_2}, \cdots, \frac{\partial J}{\partial w_N} \Biggr) $$

$$ W := W - \alpha \nabla J(W) $$

위의 공식은 일반적인 경사 하강에 대한 표준 공식입니다. 선형 회귀 분석 또는 모든 현실적인 선형 최적화 문제에 대한 최적의 매개 변수 집합을 얻을 수 있습니다. 만약 여러분이 이 공식의 중요성을 이해한다면, 여러분은 신경망이 어떻게 훈련되는지 "간단히" 이해할 것입니다. 하지만 실제로는 신경망의 훈련 과정을 복잡하게 만드는 어떤 것들이 있고, 다음 절에서 그것들을 다루는 방법에 대해 다루도록 하겠습니다.

# 신경망에 경사 하강법 적용하기

## 볼록하다는 것의 문제

이전 섹션에서는 단순한 선형 회귀 문제에 대해 경사 하강을 실행하는 방법을 보여 주었고, 그렇게 하면 올바른 매개 변수를 찾을 수 있다고 선언했습니다. 이것은 우리가 했던 것처럼 선형 모델을 최적화하는 것은 사실이지만, 활성화 함수가 가지고 있는 비선형성 때문에 신경망에는 적용되지 않습니다. 결과적으로, 신경망의 손실 함수는 '그릇 모양'이 아니고, 볼록하지 않습니다. 대신, 수 많은 언덕과 계곡, 곡선 및 기타 불규칙성으로 인해 손실 함수가 훨씬 더 복잡합니다. 즉, 손실값이 주변에서는 가장 낮지만 반드시 절대 최소값(또는 "global minima")은 아닌 "극솟값(local minima)"이 많다는 의미입니다. 이것은 우리가 경사 하강을 할 때, 우리는 우연히 극솟값에 갇힐 수 있다는 것을 의미합니다.

{% include figure_multi.md path1="/images/figures/non_convex_function.png" caption1="두 개의 매개 변수가 있는 볼록하지 않은 손실 함수면의 예입니다. 심층 신경망에서는 수백만 개의 매개 변수를 다루고 있지만 기본 원리는 그대로입니다. 출처: <a href=\"http://videolectures.net/site/normal_dl/tag=983679/deeplearning2015_bengio_theoretical_motivations_01.pdf\">Yoshua Bengio</a>." %}

이 책의 범위를 벗어난 어떤 이론적 이유로 인해 이것은 딥 러닝에서 큰 문제가 아닌 것으로 밝혀졌습니다. 왜냐하면 다른 기준들과 함께 충분한 숨겨진 단위가 있을 때, 대부분의 극솟값은 합리적으로 절대 최소값에 가깝기 때문에 "충분히" 좋기 때문입니다. [Dauphin et al](https://arxiv.org/abs/1406.2572)에 따르면, 극솟값보다 더 큰 문제는 기울기가 0에 매우 근접하는 [안장점](https://ko.wikipedia.org/wiki/안장점),입니다. 이것이 사실인 이유에 대한 설명은 [요슈아 벤지오](http://www.iro.umontreal.ca/~bengio/yoshua_en/)의 [강의](http://videolectures.net/deeplearning2015_bengio_theoretical_motivations/)를 참고하세요(28절, 1:09:41)

극솟값이 큰 문제가 아니라는 사실에도 불구하고, 우리는 여전히 그것들이 전혀 문제가 되지 않도록 극복하는 것을 선호합니다. 한 가지 방법은 경사 하강이 작동하는 방식을 수정하는 것입니다. 다음 절에서는 이 방법을 설명합니다.

## 확률적, 배치 그리고 미니 배치 경사 하강

극솟값 외에도, "기본" 경사 하강은 또 다른 큰 문제가 있습니다: 너무 느립니다. 신경망은 수억 개의 매개 변수를 가질 수 있습니다. 즉, 데이터 세트의 단일 예제를 평가하려면 수억 개의 작업이 필요합니다. 게다가 데이터 세트의 모든 지점에서 평가된 경사 하강("batch gradient downlation")은 매우 비싸고 느린 작업입니다. 더욱이, 모든 데이터 세트에는 고유의 중복성이 있기 때문에, 어쨌든 점들의 충분히 많은 부분 집합에서 전체의 경사를 예상할 수 있어, 기울기를 추정하기 위한 부분 경사 하강에 불필요한 비용이 많이 든다는 것을 알 수 있습니다.

우리는 [확률적 경사 하강법(SGD)](https://en.wikipedia.org/wiki/Stochastic_gradient_descent)라는 수정된 경사 하강법을 사용하여 이 문제와 극솟값 문제를 모두 해결할 수 있습니다. SGD를 사용하여 데이터 세트를 섞은 다음 각 샘플을 개별적으로 살펴보고, 단일 점에 대한 기울기를 계산하고, 각 샘플에 대한 가중치 업데이트를 수행합니다. 단일 예제가 특이치일 수 있고 실제 기울기의 근사치가 반드시 좋은 것은 아니기 때문에 처음에는 좋지 않은 생각처럼 보일 수 있습니다. 그러나 데이터 세트의 각 샘플에 대해 임의의 순서로 이 작업을 수행하면 전체 기울기 업데이트 경로의 변동이 평균화되고 좋은 해로 수렴됩니다. 게다가 SGD는 업데이트를 더 "덜컹거리고" 불규칙하게 만들어 극솟값과 안장점을 벗어나 계곡의 바닥에 갇히지 않도록 도와줍니다.

SGD는 특히 손실 함수면이 불규칙한 경우에 유용합니다. 그러나 일반적으로는 전체 데이터 세트가 각각 $$K$$개의 샘플을 갖도록 동일한 크기로 나누어진 $$N$$개의 미니 배치를 사용하는 미니 배치 경사 하강법(MB-GD)을 사용하는 것이 일반적인 접근 방식이다. $$K$$는 적은 양의 양수이거나 수십 또는 수백이 될 수도 있습니다. 특정 아키텍처와 애플리케이션에 따라 다릅니다. $$K=1$$이면 SGD이고 $$K$$가 전체 데이터 세트의 크기라면 배치 경사 하강입니다. 혼란스럽게도, 때때로 사람들은 MB-GD와 한 번에 하나의 표본을 모두 참조하기 위해 "SGD"라고 말합니다.

MB-GD를 사용하면 다음 두 가지 모두를 최대한 활용할 수 있습니다. 기울기는 SGD보다 부드럽고 안정적이며 전체 기울기와 상당히 유사할 뿐 아니라, 각 업데이트에 대해 데이터 세트의 모든 샘플을 평가할 필요가 없기 때문에 굉장히 빠릅니다. MB-GD는 병렬 가능한 행렬 연산으로 인해 매우 효율적으로 계산됩니다.

{% include figure_multi.md path1="/images/figures/bumpy_gradient_descent.png" caption1="볼록하지 않은 손실 함수에 대한 경사 하강법 예시(신경망과 같은), $\theta_0$ 와 $\theta_1$, 2개의 매개 변수가 있습니다. Source: <a href=\"http://www.holehouse.org/mlclass/01_02_Introduction_regression_analysis_and_gr.html\">Andrew Ng</a>." %}

실제로 MB-GD와 SGD는 신경망의 손실 함수를 효율적으로 최적화하는 데 효과적입니다. 하지만, 그들은 약점도 가지고 있습니다.

 - 앞서 언급한 안장점의 문제. 손실 함수 기울기가 0에 매우 근접하는 매개변수에 갇힐 수 있습니다.
 - 학습률은 수동으로 설정해야 하는 하이퍼 파라미터로 유지되며, 이는 어려운 작업입니다. 학습률이 너무 작으면 수렴 속도가 느려지고, 너무 크면 올바른 경로를 벗어나 오버슈팅할 수 있습니다.

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

