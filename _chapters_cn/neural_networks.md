---
layout: chapter
title: "神经网络"
date: 2016-01-04
---

Nearly a century before neural networks were first implemented, [Ada Lovelace](http://findingada.com/) described an ambition to build a "[calculus of the nervous system](http://www.thelancet.com/journals/lancet/article/PIIS0140-6736(15)00686-8/fulltext?rss=yes)." Although speculative analogies between brains and machines are as old as the philosophy of computation itself, it wasn't until [Charles Babbage](https://en.wikipedia.org/wiki/Charles_Babbage) proposed the [Analytical engine](https://en.wikipedia.org/wiki/Analytical_Engine) that we conceived of "calculators" having humanlike cognitive capacities. Ada would not live to see her dream of building the engine come to fruition, as engineers of the time were unable to produce the complex circuitry her schematics required. Nevertheless, the idea was passed on to the next century when [Alan Turing](https://en.wikipedia.org/wiki/Alan_Turing) cited it as the inspiration for the [Imitation Game](http://phil415.pbworks.com/f/TuringComputing.pdf), what soon came to be called the "[Turing Test](https://en.wikipedia.org/wiki/Turing_test)." His ruminations into the extreme limits of computation incited the first boom of artificial intelligence, setting the stage for the golden age of neural networks.

在神经网络最初实现的一个世纪前，[Ada Lovelace](http://findingada.com/) 就有建立一个“[神经系统的微积分](http://www.thelancet.com/journals/lancet/article/PIIS0140-6736(15)00686-8/fulltext?rss=yes)”的野心，尽管将大脑同机器类比和计算哲学一样历史悠久，但直到 [Charles Babbage](https://en.wikipedia.org/wiki/Charles_Babbage) 提出[分析机(Analytical engine)](https://en.wikipedia.org/wiki/Analytical_Engine)之后，我们才开始考虑计算器有类似人类的认知能力。Ada 并没在有生之年看到从建造引擎到产生成果的过程，因为在那时的工程师也无法把她要求的复杂电路实现出来。然而，这个想法传到了下一个世纪，图灵以之为灵感创造了[模仿游戏(Imitation Game)](http://phil415.pbworks.com/f/TuringComputing.pdf)，这就是后来大名鼎鼎的[图灵测试(Turing Test)](https://en.wikipedia.org/wiki/Alan_Turing)。他关于计算之极限的思考被视为人工智能最初的一现昙华，为神经网络的黄金时代奠定了基础。


## The once and future king 曾经与未来之王

The recent resurgence of neural networks is a peculiar story. Intimately connected to the early days of AI, neural networks were first formalized in the late 1940s in the form of Turing's [B-type machines](https://en.wikipedia.org/wiki/Unorganized_machine), drawing upon earlier research into [neural plasticity](https://en.wikipedia.org/wiki/Hebbian_theory) by neuroscientists and cognitive psychologists studying the learning process in human beings. As the mechanics of brain development were being discovered, computer scientists experimented with idealized versions of action potential and neural backpropagation to simulate the process in machines. 

近来神经网络的复兴是一个很特别的故事。它和早期的人工智能密切相关，在 20 世纪 40 年代末期，借鉴了神经科学家和认知心理学家对人类学习过程中[神经可塑性](https://en.wikipedia.org/wiki/Hebbian_theory)的早期研究，神经网络在图灵的 [B 类机器(B-type machines)](https://en.wikipedia.org/wiki/Unorganized_machine)中首次正式公布。随着大脑发育机制的发现，计算机科学家实验了理想状态下的动作电位（action potential）和神经反向传播（neural backpropagation），试图在机器中模拟这个过程。

Today, most scientists caution against taking this analogy too seriously, as neural networks are strictly designed for solving machine learning problems, rather than accurately depicting the brain. Nevertheless, the metaphor of the core unit of neural networks as a simplified biological neuron has stuck over the decades. The progression from biological neurons to artificial ones can be summarized by the following figures.

今天，大多数科学家警示我们不要过度使用类比，因为神经网络仅是被设计来解决机器学习的问题的，并非精确描绘人脑。然而，把神经网络核心单元比作简化的生物神经元的隐喻已深植人们的脑海。生物神经元到人工神经元的发展可以总结为下述图像。

<style>
#outer {
	text-align: center;
	margin-left:-300px;
	margin-right:-300px;
	display:inline-block;
	padding:20px;
}
.insert {
	display: inline-block;
	margin-left:5px;
	margin-right:5px;
	border: 1px solid #ddd;
	padding:5px;
}
.caption {	
	line-height:150%;
	color:#666;
	background-color:#f4f4f4;
	margin-top:8px;
}

</style>
<center>
<div id="outer">
	<div class="insert"><img src="/images/neuron-anatomy.jpg" />
		<div class="caption">
			Anatomy of a biological neuron
			<br/>Source: <a href="https://askabiologist.asu.edu/neuron-anatomy">ASU school of life sciences</a>
		</div>
	</div>
	<div class="insert">
		<img src="/images/neuron-simple.jpg" />
		<div class="caption">
			Simplified neuron body within a network
			<br/>Source: <a href="http://www.generation5.org/content/2000/nn00.asp">Gurney, 1997. An Introduction to Neural Networks</a>
		</div>
	</div>
	<div class="insert">
		<img src="/images/neuron-artificial.png" />
		<div class="caption">
			Artificial neuron (<b>fix this</b>)
			<br/>&nbsp;
		</div>
	</div>	
</div>
</center>


Neural networks took a big step forward when [Frank Rosenblatt](https://en.wikipedia.org/wiki/Frank_Rosenblatt) devised the [Perceptron](https://en.wikipedia.org/wiki/Perceptron) in the late 1950s, a type of linear classifier that we saw in the [last chapter](/2016/01/03/machine-learning.html). Publicly funded by the U.S. Navy, the Mark 1 perceptron was designed to perform image recognition from an array of photocells, potentiometers, and electrical motors. It's effectiveness at completing complex electrical circuits lead the New York Times in 1958 to predict that a machine would soon ["walk, talk, see, write, reproduce itself and be conscious of its existence"](http://query.nytimes.com/gst/abstract.html?res=9D01E4D8173DE53BBC4053DFB1668383649EDE).
Frank Rosenblatt


[Frank Rosenblatt](https://en.wikipedia.org/wiki/Frank_Rosenblatt) 在 20 世纪 50 年代后期发明了[感知器（Perceptron）](https://en.wikipedia.org/wiki/Perceptron)，也就是我们在[上一章](/2016/01/03/machine-learning.html)看到的线性分类器，这对于神经网络是很大的进步。被美国海军公开资助的 Mark 1 感知器是用一串光电管，电位器，和电机来实现图像识别功能。它能高效地执行复杂的电路操作，以至于纽约时报在 1958 年预测这个机器不久就能“[走路，说话，观察，书写，自我复制和意识到自身的存在](http://query.nytimes.com/gst/abstract.html?res=9D01E4D8173DE53BBC4053DFB1668383649EDE)”。

The early hype would inspire science fiction writers for decades to come, but the excitement was far more tempered in the academic community. Marvin Minsky's and Seymour Papert's 1969 book, [Perceptrons](https://en.wikipedia.org/wiki/Perceptrons_(book)), demonstrated various-—[even trivial](http://users.ecs.soton.ac.uk/harnad/Hypermail/Explaining.Mind96/0140.html)—limitations, inadvertently leading to a [decline of interest](https://en.wikipedia.org/wiki/AI_winter) within both academia and the general public, who had mistakenly assumed computers would simply keep up with the [breakneck pace](https://en.wikipedia.org/wiki/Moore%27s_law) of computational power. Even Turing himself said machines would possess human-level intelligence by the year 2000 -- the year we had the [Y2K scare](https://en.wikipedia.org/wiki/Year_2000_problem). 

之前的大肆宣传使得科幻作家们相信这在未来几十年就会成真，但学术圈远比他们淡定。Marvin Minsky 和 Seymour Papert 1969 年的文章 [感知器（Perceptrons）](https://en.wikipedia.org/wiki/Perceptrons_(book))中，描绘了许多——[即便是非常琐碎的](http://users.ecs.soton.ac.uk/harnad/Hypermail/Explaining.Mind96/0140.html)——限制，不经意间导致公众和学术圈对人工智能的[兴趣减弱](https://en.wikipedia.org/wiki/AI_winter)，他错误地假设电脑的计算能力会继续以 [极快而危险的速度增长](https://en.wikipedia.org/wiki/Moore%27s_law)。即便是图灵自己也预言在 2000 年机器会具备类人的智力水平——恰好在[千年虫问题](https://en.wikipedia.org/wiki/Year_2000_problem)引起恐慌的这一年。

Despite a number of quiet but significant improvements to neural networks in the 80s and 90s [[1]](_jurgen_)[[2]](_)[[3]](_Perceptrons_), they remained on the sidelines through the 2000s, with most commercial and industrial applications of machine learning favoring [support vector machines](https://en.wikipedia.org/wiki/Support_vector_machine) and various other approaches. [Starting in 2009](http://www.cs.utoronto.ca/~gdahl/papers/dbnPhoneRec.pdf) and [especially ramping up from 2012](https://www.technologyreview.com/s/530561/the-revolutionary-technique-that-quietly-changed-machine-vision-forever/), neural networks have once again become the dominant strain of ML algorithms. Their resurgence was largely brought about by the emergence of [convolutional](/2016/02/02/convnets.html) and [recurrent neural networks](/2016/02/10/RNNs.html), which have surpassed (sometimes dramatically so) previous state-of-the-art methods for key problems in the audiovisual domain. But more interestingly, they have a number of new applications and properties not seen before, especially of a kind that has piqued the interest of artists and others from outside the AI field proper. This book will look more closely at convolutional neural networks in particular several chapters from now.

尽管在 80 和 90 年代神经网络有一系列不显眼但重大的进步[[1]](_jurgen_)[[2]](_)[[3]](_Perceptrons_)，它们在 2000 年初时都处在二线，而大多数商业和工业中的机器学习应用都偏好[支持矢量机（support vector machines）](https://en.wikipedia.org/wiki/Support_vector_machine)和许多其他手段。[从 2009 年](http://www.cs.utoronto.ca/~gdahl/papers/dbnPhoneRec.pdf)，尤其是[ 2012  年开始](https://www.technologyreview.com/s/530561/the-revolutionary-technique-that-quietly-changed-machine-vision-forever/)，神经网络重新成为机器学习算法的主旋律。神经网络的复兴很大程度上是由于[卷积神经网络](/2016/02/02/convnets.html)和 [递归神经网络](/2016/02/10/RNNs.html)的出现所带来的，它们在解决视觉和听觉的关键问题时，显著地（甚至是戏剧性地）超过了之前最厉害的方法。但更有趣的是，它们有许多之前没有见过的新应用和属性，尤其是一些应用完全激起了艺术家们和各种圈子的人的兴趣。这本书会在现在开始的几个章节详细讲到卷积神经网络。

Although many learning algorithms have been proposed over the years, we will mostly focus our attention on neural networks because:

尽管这些年提出了许多机器学习算法，我们还是主要学习神经网络，是因为：

 - They have a surprisingly simple and intuitive formulation.
 
 - 它们的结构异常简单而符合直觉。

 - Deep neural networks are **tue？** current state-of-the-art in several important machine learning tasks, the ones most relevant to this book.
 
 - 深度神经网络与目前最先进的，和本书相关的几个机器学习任务有关。
 
 - Most of the recent creative uses of machine learning have been made with neural networks.
 
 - 近来大多数机器学习的创意应用都是用神经网络来制作的。



## From linear classifiers to neurons 从线性分类器到神经元

Recall from the previous chapter that the input to a 2d linear classifier or regressor has the form:

回想一下之前的章节，2d 线性分类器的输入值或者说回归量长这样：

$$
\begin{eqnarray}
f(x_1, x_2) = b + w_1 x_1 + w_2 x_2
\end{eqnarray}
$$

More generally, in any number of dimensions, it can be expressed as

推广一下，到任意维度中，可以表示为：

$$
\begin{eqnarray}
f(X) = b + \sum_i w_i x_i
\end{eqnarray}
$$

In the case of regression, $$f(X)$$ gives us our predicted output, given the input vector $$X$$. In the case of classification, our **predicted class？** is given by 

在回归的例子中，输入向量 $$X$$，预测输出为 $$f(X)$$。在分类器的例子中，我们预测的分类可由下得出：

$$
\begin{eqnarray}
  \mbox{classification} & = & \left\{ \begin{array}{ll}
  	  1 & \mbox{if } f(X) \gt 0 \\
      0 & \mbox{if } f(X) \leq 0
      \end{array} \right.
\tag{1}\end{eqnarray}
$$

Each weight, $$w_i$$, can be interpreted as signifying the relative influence of the input that it's multiplied by, $$x_i$$. The $$b$$ term in the equation is often called the _bias_, because it controls how predisposed the neuron is to firing a 1 or 0, irrespective of the weights. A high bias makes the neuron require a larger input to output a 1, and a lower one makes it easier.

每个权重 $$w_i$$ 代表和它相乘的输入值 $$x_i$$ 的相对影响。公式中的 $$b$$ 通常叫做偏置（ bias），因为它控制神经元多大程度倾向于发射 1 或者 0，不论它的权重是多少。高的偏置意味着神经元需要更大的输入来输出 1，而低的则不需要那么大。

We can get from this formula to a full-fledged neural network by introducing two innovations. The first is the addition of an _activation function_, which turns our linear discriminator into what's called a _neuron_, or a "_unit_" (to dissociate them from the brain analogy). The second innovation is an architecture of neurons which are connected sequentially in _layers_. We will introduce these innovations in that order.

我们只要再加入两处创新，就可以从这个公式中推导出完备的神经网络。第一个是加上一个**激活函数**（activation function），它会把我们的线性鉴别器变成所谓的“神经元”，或者“单元”（与人脑模拟区分开）。第二个创新点在于在“层”（layers）中依序连接的神经元的结构。我们会一个一个介绍到这些创新。


## Activation function 激活函数

In both artificial and biological neural networks, a neuron does not just output the bare input it receives. Instead, there is one more step, called an _activation function_, analogous to the rate of [action potential](https://en.wikipedia.org/wiki/Action_potential) firing in the brain. The activation function takes the same weighted sum input from before, $$z = b + \sum_i w_i x_i$$, and then transforms it once more before finally outputting it.

在人工和生物神经网络中，神经元都不仅是输出它接收到的输入值。中间还有一步，叫做**激活函数**（activation function），类似于动作电位向大脑发射信号的比率，激活函数和之前输入的总和有相同的权重，$$z = b + \sum_i w_i x_i$$，并且紧接着将它再转换一次，再最终输出。

同样在人工和生物神经网络中，一个神经元不会只输出它所接收到的信号。所代替的，是还有另外一步，叫做激活函数，你可以把它和大脑中的潜在行为激活.这个激活函数和之前输入的数量相同，$$z = b + \sum_i w_i x_i$$，然后在最终输出前会再次转化。

Many activation functions have been proposed, but for now we will describe two in detail: sigmoid and ReLU. 

人们提出了许多种激活函数，但是我们现在只详细描述两种：sigmoid 和 ReLU。

Historically, the [sigmoid](https://en.wikipedia.org/wiki/Sigmoid_function) function is the oldest and most popular activation function. It is defined as:

历史上，sigmoid 函数是最老且最为流行的激活函数。它的定义是：

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

$$e$$ denotes the [exponential constant](https://en.wikipedia.org/wiki/E_(mathematical_constant)), roughly equal to 2.71828. A neuron which uses a sigmoid as its activation function is called a _sigmoid neuron_. We first set the variable $$z$$ to our original weighted sum input, and then pass that through the sigmoid function.

$$e$$代表[指数常数](https://en.wikipedia.org/wiki/E_(mathematical_constant))，约等于2.71828。使用 sigmoid 作为激活函数的神经元叫做 sigmoid 神经元。我们首先令变量 $$z$$ 等于初始输入值的加权和，然后把它放入 sigmoid 函数中进行运算。

$$
z = b + \sum_i w_i x_i \\
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

At first, this equation may seem complicated and arbitrary, but it actually has a very simple shape, which we can see if we plot the value of $$\sigma(z)$$ as a function of the input $$z$$. 

一开始，这个方程可能看起来很复杂且任意，但是它其实有一个非常简单的外形。如果我们把输入值 $$z$$ 换成  $$\sigma(z)$$ 就会看得到。

{:.center}
![sigmoid](/images/sigmoid.png 'sigmoid')

We can see that $$\sigma(z)$$ acts as a sort of "squashing" function, condensing our previously unbounded output to the range 0 to 1. In the center, where $$z = 0$$, $$\sigma(0) = 1/(1+e^{0}) = 1/2$$. For large negative values of $$z$$, the $$e^{-z}$$ term in the denominator grows exponentially, and $$\sigma(z)$$ approaches 0. Conversely, large positive values of $$z$$ shrink $$e^{-z}$$ to 0, so $$\sigma(z)$$ approaches 1.

我们可以看到 $$\sigma(z)$$ 的效果类似于一种“压缩函数”，把之前无界的输出值的范围压缩为 0 到 1。在中心，当$$z = 0$$， $$\sigma(0) = 1/(1+e^{0}) = 1/2$$。当 $$z$$ 等于很大的负数，分母中的 $$e^{-z}$$ 呈几何级数增长，此时 $$\sigma(z)$$ 趋近于 0。相反地，当 $$z$$ 等于很大的正数，$$e^{-z}$$ 趋近于 0，此时  $$\sigma(z)$$ 趋近于 1。

The sigmoid function is continuously differentiable, and its derivative, conveniently, is $$\sigma^\prime(z) = \sigma(z) (1-\sigma(z))$$. This is important because we have to use calculus to train neural networks, but don't worry about that for now.

sigmoid 函数是连续可微的，而它的导数，很简单地，是 $$\sigma^\prime(z) = \sigma(z) (1-\sigma(z))$$。这个很重要因为我们要用微积分来训练神经网络，但现在还不用担心这些。

Sigmoid neurons were the basis of most neural networks for decades, but in recent years, they have fallen out of favor. The reason for this will be explained in more detail later, but in short, they make neural networks that have many layers difficult to train due to the [损失梯度问题（vanishing gradient problem）](https://en.wikipedia.org/wiki/Vanishing_gradient_problem). Instead, most have shifted to using another type of activation function, the _rectified linear unit_, or ReLU for short. Despite its obtuse name, it simply given by $$R(z) = max(0, z)$$.

过去几十年中，sigmoid 神经元是大多数神经网络的基础，但是近几年它已然失宠。原因我们稍后详解，简单来说就是，人们创造了有很多层的神经网络，因为[vanishing gradient problem](https://en.wikipedia.org/wiki/Vanishing_gradient_problem)很难用 sigmoid 方法来训练神经网络。因而，大多数转而使用其他激活函数，修正线性单元（rectified linear unit），简称为 ReLU。尽管名字有点拗口，它的公式简单表述为 $$R(z) = max(0, z)$$。


{:.center}
![ReLU](/images/relu.png 'ReLU')

In other words, ReLUs let all positive values pass through unchanged, but just sets any negative value to 0. Although newer activation functions are gaining traction, most deep neural networks these days use ReLU or one of its [closely related variants](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)). 

换句话说，ReLUs 令所有的正数保持原样，但是会把所有的负数变为 0。尽管新的激活函数正在发展，大多数深度神经网络还是在用 ReLU 或者它的一个[很接近的变体](https://en.wikipedia.org/wiki/Rectifier_(neural_networks))。

Regardless of which activation function is used, we can visualize a single neuron with this standard diagram, giving us a nice intuitive visual representation of a neuron's behavior.

不论用的是哪种激活函数，我们都可以用这个标准图来可视化一个神经元，它直观地展示了一个神经元的行为。

{:.center}
![ReLU](/images/neuron.png 'Neuron')

The above diagram shows a neuron with three inputs, and outputs a single value $$y$$. As before, we first compute the weighted sum of its inputs, then pass it through an activation function $$\sigma$$.

上述图表展示了一个有三个输入值，一个输出值  $$y$$ 的神经元。一如既往，我们先计算输入值的加权总和，然后令这个值通过激活函数 $$\sigma$$。

$$
z = b + w_1 x_1 + w_2 x_2 + w_3 x_3 \\
y = \sigma(z)
$$

You may be wondering what the purpose of an activation function is, and why it is preferred to simply outputting the weighted sum, as we do with the linear classifier from the last chapter. The reason is that a weighted sum is [_linear_](https://en.wikipedia.org/wiki/Linearity) with respect to its inputs, **i.e. it has a flat dependence on each of the inputs.** In contrast, non-linear activation functions greatly expand our capacity to model curved or otherwise non-trivial functions. This will become clearer in the next section.

你可能很好奇激活函数的目的是什么，为什么我们在上一章玩线性分类器的时候，它只是简单输出了加权总和。原因就是加权总和只是几个输入值的[线性（linear）](https://en.wikipedia.org/wiki/Linearity)加总。也就是说，加权和与每一个输入值的相关性是单调的。相比之下，非线性激活函数极大地扩展了我们模拟曲线或其他重要函数的能力。在接下来的部分中，这一点会被解释得更清楚。



# Layers 神经层

Now that we have described neurons, we can now define neural networks. A neural network is composed of a series of _layers_ of neurons, such that all the neurons in each layer connect to the neurons in the next layer. 

我们已经讲过了神经元，现在可以给出神经网络的定义了。一个神经网络是由一系列层层（layers）排列的神经元组成的，每一层的神经元都与相邻层级的神经元相关联。

{:.center}
![neural network](/images/network.png 'neural network')

Note that when we count the number of layers in a neural network, we only count the layers with connections flowing into them (omitting our first, or _input layer_). So the above figure is of a 2-layer neural network with 1 _hidden layer_. It contains 3 input neurons, 2 neurons in its hidden layer, and 1 output neuron.

注意当我们在计算一个神经网络中有多少神经层的时候，我们只计算那些有连接流入它们的那些神经层（略掉第一，或者说是输入层（input layer））。因此，上图中应该有两层神经网络并且其中之一是隐层（hidden layer）。它包含三个输入神经元，两个神经元在隐层，还有一个输出神经元。

Our computation starts with the input layer on the left, from which we pass the values to the hidden layer, and then in turn, the hidden layer will send its output values to the last layer, which contains our final value.

我们从左侧的输入层开始计算，数值从这里传给隐层，隐层再将其输出的数值传给最后一层，最后一层存有最终值。

Note that it may look like the three input neurons send out multiple values because each of them are connected to both of the neurons in the hidden layer. But really there is still only one output value per neuron, it just gets copied along each of its output connections. Neurons always output one value, no matter how many subsequent neurons it sends it to.

注意，这三个输入神经元与隐层的两个神经元之间都有连接，因此看起来好像是它们传输了多个数值。但实际上，每个神经元都只输出了一个数值，这个数值只是被复制到每个传出连接中。不论有多少输出对象，神经元们都总是只输出一个数值。



# Forward propagation 正向传播

The process of a neural network sending an initial input forward through its layers to the output is called _forward propagation_ or a _forward pass_ and any neural network which works this way is called a _feedforward neural network_. As we shall soon see, there are some neural networks which allow data to flow in circles, but let's not get ahead of ourselves yet... 

一个神经网络将最初的输入值经过其神经层并最终得到输出值的过程叫做正向传播（forward propagation 或 forward pass），任何以这种方式工作的神经网络都可以被称为前馈神经网络（feed forward neural network）。

Let's demonstrate a forward pass with this interactive demo. Click the 'Next' button in the top-left corner to proceed. You can see a forward pass in action in the following demo. 

我们用这个可交互 demo 来展示一下正向传播。按下左上角的 “Next” 键来继续。你将会看到一个正向传播案例。

{:.center}
![neural network](/images/temp_demo_forward_pass.png 'forward_pass')



# More layers, more expressiveness 更多神经层，更强的表现力

Why are hidden layers useful? The reason is that if we have no hidden layers and map directly from inputs to output, each input's contribution on the output is independent of the other inputs. In real-world problems, input variables tend to be highly interdependent and they affect the output in combinatorially intricate ways. The hidden layer neurons allow us to capture subtle interactions among our inputs which affect the final output downstream. 

为什么隐层非常有用？原因是如果我们没有隐层，直接从输入到输出，每个输入之于输出的作用就会完全独立于其他的输入。在现实世界的问题中，输入变量总是倾向于高度相互依存的，同时他们对于输出的影响也是组合的、复杂的。隐层神经元让我们能够获取输入值之间那些微妙的互动，这些互动会一层层向下影响最终的输出值。

Another way to interpret this is that the hidden layers represent higher-level "features" or attributes of our data. Each of the neurons in the hidden layer weigh the inputs differently, learning some different intermediary characteristic of the data, and our output neuron is then a function of these instead of the raw inputs. By including more than one hidden layer, we give the network an opportunity to learn multiple levels of abstraction of the original input data before arriving at a final output. This notion of high-level features will become more concrete in the next chapter when we look closely at the hidden layers.

用另一种方式来解释就是，隐层代表了这些数据更高级别的“特征”或者说属性。每一个隐层神经元都给输入值加了不同的权重，学习数据的不同中间特征，然后我们的输出神经元将会处理这些数据，而不是原始的未处理的输入数据。当有多个隐层时，神经网络就有机会在达到最终输出之前，学习到原始输入数据多层抽象。下一章我们会对隐层作更深入的了解，到时候高级别特征这个概念到会更加清晰。


Recall also that activation functions expand our capacity to capture non-linear relationships between inputs and outputs. By chaining multiple non-linear transformations together through layers, this dramatically increases the flexibility and expressiveness of neural networks. The proof of this is complex and beyond the scope of this book, but it can even be shown that any 2-layer neural network with a non-linear activation function (including sigmoid or ReLU) is a [_universal function approximator_](http://www.sciencedirect.com/science/article/pii/0893608089900208), that is it's theoretically capable of expressing any arbitrary input-to-output mapping. This property is what makes neural networks so powerful.

回想一下，激活函数也对我们发掘输入值和输出值之间的非线性关系非常有帮助。通过将多重非线性转换关系通过神经层串联在一起，神经网络的灵活性和表现力得到了极大程度地提升。其证明过程非常复杂，并且超出了本书的讲述范围，但这甚至能够说明任何具有非线性激活函数（包括sigmoid 或 ReLU)）的双层神经网络都是一个[万能函数逼近器（universal function approximator）](http://www.sciencedirect.com/science/article/pii/0893608089900208)，理论上能够表达任意由输入到输出的映射。而正是这一特质使得神经网络如此强大。



# Regression and classification 回归与分类

In the demo above, we map a set of inputs to a single output in a forward pass. We can interpret the forward pass as making a prediction about the output, given the input. This is called _regression_ and is one of the primary uses of neural networks. Let's try an example.

在上面的 demo 中，我们在一个正向传播过程中，将一组输入值映射到单独的输出值。我们可以把正向传播比作给定输入值，预测输出值。这就叫做**回归**，这也是神经网络的基础运用之一。我们来用一个实例解释一下。



-----

[[ THIS IS A DRAFT]]
Random set of dat, 3 cols. 1 regression value

Interactive 3 -> 1

Measure the error. Use L2 error

Suppose we are trying to model ___...

We have five observed examples of data.

Give random weights. Not very accurate.

Now a magic trick, I'll give it a new set of weights. Now let's run the examples again, and see that they are all correct now. 

the process of obtaining the correct weights is called training. for now, ignore it, it's a black box. we'll talk about it in the next section.


-----


What about classification? In the previous chapter, we introduced binary classification by simply thresholding the output at 0; If our output was positive, we'd classify positively, and if it was negative, we'd classify negatively. For neural networks, it would be reasonable to adapt this approach for the final neuron, and classify positively if the output neuron scores above some threshold. For example, we can threshold at 0.5 for sigmoid neurons which are always positive.

分类是什么？在之前的章节中，我们介绍了二元分类，通过简单地设一个阀值，令未通过的输出为 0；如果我们的输出值是正的，我们就正着分类，如果输出是负的，我们就负着分类。对神经网络来说，用这种方法一路逼近最终的神经元是可行的，并且如果输出神经元大于某个阀值就正着分类。比如说，我们可以给 sigmoid  神经元设阀值 0.5，因为它的输出始终是正数。

But what if we have multiple classes? One option might be to create intervals in the output neuron which correspond to each class, but this would be problematic for reasons that we will learn about when we look at [how neural networks are trained](). Instead, neural networks are adapted for classification by having one output neuron for each class. We do a forward pass and our prediction is the class corresponding to the neuron which received the highest value. Let's have a look at an example.

但如果我们有很多类呢？一个选择可能是给输出神经元设几个区间，每个对应不同的分类，但是这个方式可能会很麻烦，当我们学习[神经网络是如何训练的]()的时候就知道了。相比之下，神经网络特别适合于这样分类：给每个类设一个输出神经元。当我们做一个正向传播时，我们的预测结果就是得到输出值最高的那个神经元所代表的分类。让我们来看一个例子。



# Classification of handwritten digits 手写体数字的分类

Let's now tackle a real world example of classification using neural networks, the task of recognizing and labeling images of handwritten digits. We are going to use the [MNIST dataset](http://yann.lecun.com/exdb/mnist/), which contains 60,000 labeled images of handwritten digits sized 28x28 pixels, whose classification accuracy serves as a common benchmark in machine learning research. Below is a random sample of images found in the dataset.

现在让我们来解决一些真实世界中，使用神经网络进行分类的例子：识别和标记手写数字的图像。我们将会用到 [MNIST 数据集](http://yann.lecun.com/exdb/mnist/)，这个数据集有 60,000 标记过的手写数字图像，每个大小 28x28 像素。能多大程度对这个数据集精确分类是机器学习研究通用的一个基准评效测试。下面是一些数据集中随机的样本图片。

{:.center}
![MNIST ground truth](/images/temp_fig_mnist_groundtruth.png 'MNIST ground truth')

The way we setup a neural network to classify these images is by having the raw pixel values be our first layer inputs, and having 10 output classes, one for each of our digit classes from 0 to 9. Since they are grayscale images, each pixel has a brightness value between 0 (black) and 255 (white). All the MNIST images are 28x28, so they contain 784 pixels. We can unroll these into a single array of inputs, like in the following figure.

我们要搭一个神经网络来对这些图像进行分类，方法是，在第一层输入层给出初始的图像像素数据，然后设定 10 个输出类别，分别是从 0 到 9。因为这些都是灰度图像，每个像素都有从 0（黑） 到 255（白） 的亮度值。所有的 MNIST 都是 28x28，所以它们每个有 784 个像素。我们可以把 784 个像素的数据展开成一个数列，依次输入，如下图所示。

{:.center}
![MNIST](/images/temp_fig_mnist.png 'MNIST')

The important thing to realize is that although this network seems a lot more imposing than our simple 3x2x1 network in the previous chapter, it works exactly as before, just with many more neurons. Each neuron in the first hidden layer receives ....

最重要的事是，要注意尽管这似乎比之前章节简单的 3x2x1 网络酷炫很多，它和之前的原理是一模一样的，仅仅是多了许多神经元而已。每个第一层隐层的神经元接收到……


{:.center}
![MNIST demo](/images/temp_demo_mnist_forwardpass.png 'MNIST demo')


# Summary 总结

TBD

# Further reading 推荐阅读

 - nielsen
 - kurekenov
 
