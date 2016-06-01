---
layout: chapter
title: "神经网络"
date: 2016-01-04
---

Nearly a century before neural networks were first implemented, [Ada Lovelace](http://findingada.com/) described an ambition to build a "[calculus of the nervous system](http://www.thelancet.com/journals/lancet/article/PIIS0140-6736(15)00686-8/fulltext?rss=yes)." Although speculative analogies between brains and machines are as old as the philosophy of computation itself, it wasn't until [Charles Babbage](https://en.wikipedia.org/wiki/Charles_Babbage) proposed the [Analytical engine](https://en.wikipedia.org/wiki/Analytical_Engine) that we conceived of "calculators" having humanlike cognitive capacities. Ada would not live to see her dream of building the engine come to fruition, as engineers of the time were unable to produce the complex circuitry her schematics required. Nevertheless, the idea was passed on to the next century when [Alan Turing](https://en.wikipedia.org/wiki/Alan_Turing) cited it as the inspiration for the [Imitation Game](http://phil415.pbworks.com/f/TuringComputing.pdf), what soon came to be called the "[Turing Test](https://en.wikipedia.org/wiki/Turing_test)." His ruminations into the extreme limits of computation incited the first boom of artificial intelligence, setting the stage for the golden age of neural networks.

在神经网络被首次实行的一个世纪前， [Ada Lovelace](http://findingada.com/)描述了一个建立“ 神经系统的微积分网络”，“尽管在大脑和机器中的投机类比和计算机哲学一样历史悠久， 直到[Charles Babbage](https://en.wikipedia.org/wiki/Charles_Babbage)提议研究机器后，我们才把微积分视作有类人脑的识别能力。Ada 并没在有生之年看到从建造引擎到果实化的过程，因为在那时的工程师也无法把她要求的复杂电路实现出来。然而，当这个想法被图灵引用作为模仿游戏灵感并传到下一个世纪，并马上被称为[图灵测试](https://en.wikipedia.org/wiki/Alan_Turing)。他对于电脑化极度缺陷的反思也激起了人工智能的初次崛起，为神经网络的黄金时代奠定了基础。
## The once and future king

The recent resurgence of neural networks is a peculiar story. Intimately connected to the early days of AI, neural networks were first formalized in the late 1940s in the form of Turing's [B-type machines](https://en.wikipedia.org/wiki/Unorganized_machine), drawing upon earlier research into [neural plasticity](https://en.wikipedia.org/wiki/Hebbian_theory) by neuroscientists and cognitive psychologists studying the learning process in human beings. As the mechanics of brain development were being discovered, computer scientists experimented with idealized versions of action potential and neural backpropagation to simulate the process in machines. 

进来神经网络的兴起是一个特别的故事。它和早期的人工智能紧密联系在一起，神经网络在图灵 [B类机器](https://en.wikipedia.org/wiki/Unorganized_machine)中被首次正式公布，作为神经科学家和认知哲学家对人类学习过程中[神经重塑性](https://en.wikipedia.org/wiki/Hebbian_theory)的早期研究，作为被发现的大脑发展机制，
计算机科学家实验了行为潜力和神经后部繁殖的理想化版本来激活机器过程。

Today, most scientists caution against taking this analogy too seriously, as neural networks are strictly designed for solving machine learning problems, rather than accurately depicting the brain. Nevertheless, the metaphor of the core unit of neural networks as a simplified biological neuron has stuck over the decades. The progression from biological neurons to artificial ones can be summarized by the following figures.

今天，大多数科学家不认同过多使用类比，因为神经网络是被严格设计来解决机器学习的问题的，而非精确描绘人脑。然而，神经网络核心单元作为简化过的神经的隐喻被误解了几十年。从生物神经元到人工神经元的进化过程，其实可以被下列首次应用的形状所总结，

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


当 [Frank Rosenblatt](https://en.wikipedia.org/wiki/Frank_Rosenblatt) 在1950s后期修改了[Perceptron](https://en.wikipedia.org/wiki/Perceptron)后，我们将会看到在[前章](/2016/01/03/machine-learning.html)中提到的线性分类器。被美国海军公共资助的M1P是从光电池，电位器，和电动马达的排列中被设计并体现照片识别功能。在完成复杂电路中的有效性领导纽约时报在1958年预测有一个机器会“走路，说话，看，写，再生和认识到自身的存在“。

The early hype would inspire science fiction writers for decades to come, but the excitement was far more tempered in the academic community. Marvin Minsky's and Seymour Papert's 1969 book, [Perceptrons](https://en.wikipedia.org/wiki/Perceptrons_(book)), demonstrated various-—[even trivial](http://users.ecs.soton.ac.uk/harnad/Hypermail/Explaining.Mind96/0140.html)—limitations, inadvertently leading to a [decline of interest](https://en.wikipedia.org/wiki/AI_winter) within both academia and the general public, who had mistakenly assumed computers would simply keep up with the [breakneck pace](https://en.wikipedia.org/wiki/Moore%27s_law) of computational power. Even Turing himself said machines would possess human-level intelligence by the year 2000 -- the year we had the [Y2K scare](https://en.wikipedia.org/wiki/Year_2000_problem). 
最早的广告宣传激励了小说作家几十年这些幻想会变成现实，但是在学术社会里激起的反应比其他地方要温和的多。Marvin Minsky和Seymour Papert1969年的文章 [感知器](https://en.wikipedia.org/wiki/Perceptrons_(book))中，描绘了许多[即使微小的](http://users.ecs.soton.ac.uk/harnad/Hypermail/Explaining.Mind96/0140.html)限制，会造成偶然性的[兴趣减少](https://en.wikipedia.org/wiki/AI_winter)，并错误地假设电脑会单纯跟上 [速度极快的](https://en.wikipedia.org/wiki/Moore%27s_law)计算能力。
即使图灵他自己也在2000年说机器会具备人类的智力－当我们具有[Y2K 等级](https://en.wikipedia.org/wiki/Year_2000_problem)的这一年。

Despite a number of quiet but significant improvements to neural networks in the 80s and 90s [[1]](_jurgen_)[[2]](_)[[3]](_Perceptrons_), they remained on the sidelines through the 2000s, with most commercial and industrial applications of machine learning favoring [support vector machines](https://en.wikipedia.org/wiki/Support_vector_machine) and various other approaches. [Starting in 2009](http://www.cs.utoronto.ca/~gdahl/papers/dbnPhoneRec.pdf) and [especially ramping up from 2012](https://www.technologyreview.com/s/530561/the-revolutionary-technique-that-quietly-changed-machine-vision-forever/), neural networks have once again become the dominant strain of ML algorithms. Their resurgence was largely brought about by the emergence of [convolutional](/2016/02/02/convnets.html) and [recurrent neural networks](/2016/02/10/RNNs.html), which have surpassed (sometimes dramatically so) previous state-of-the-art methods for key problems in the audiovisual domain. But more interestingly, they have a number of new applications and properties not seen before, especially of a kind that has piqued the interest of artists and others from outside the AI field proper. This book will look more closely at convolutional neural networks in particular several chapters from now.

尽管在80和90年代神经网络有一系列安静但显著的进步[[1]](_jurgen_)[[2]](_)[[3]](_Perceptrons_)，它们在2000年时还和工商业中使用的 [支持矢量机器](https://en.wikipedia.org/wiki/Support_vector_machine)的机器学习及其他方法一样置身事外，[从2009开始](http://www.cs.utoronto.ca/~gdahl/papers/dbnPhoneRec.pdf)且[于2012年提高](https://www.technologyreview.com/s/530561/the-revolutionary-technique-that-quietly-changed-machine-vision-forever/)，神经网络曾再次成为ML算法的主要品系。这些重现大部分是由[卷积的](/2016/02/02/convnets.html)和 [周期性神经网络](/2016/02/10/RNNs.html)的出现所带来的，它们超过了（有时是戏剧性的）之前在视听领域的关键性艺术问题。但是更有趣的是，它们具有一系列之前没有被发现过的新运用及资产，特别能激起机器艺术家及其他人工智能领域之外人的兴趣。这本书会在从现在开始的几个特别的章节中更紧密地关注卷积的神经网络。

Although many learning algorithms have been proposed over the years, we will mostly focus our attention on neural networks because:
尽管许多学习的算法被提出很多年了，我们还是把注意力放在神经网络上因为：

 - They have a surprisingly simple and intuitive formulation.
他们有非常令人惊讶的简单及有直觉力的构造。

 - Deep neural networks are tue current state-of-the-art in several important machine learning tasks, the ones most relevant to this book.
深度的神经网络的确在不同重要的机器学习案例中有真正当前先进的艺术案例，这些与本书有关的案例。
 - Most of the recent creative uses of machine learning have been made with neural networks.
机器学习最近大部分与创意有关的应用都使用神经网络来制作。



## From linear classifiers to neurons
从线性分类器到神经元
Recall from the previous chapter that the input to a 2d linear classifier or regressor has the form:
回顾之前的篇章,我们把2d线性分类器放到回归因子里就有这个形式：

$$
\begin{eqnarray}
f(x_1, x_2) = b + w_1 x_1 + w_2 x_2
\end{eqnarray}
$$

More generally, in any number of dimensions, it can be expressed as

总的来说，在所有的维度中，可以被显示为
$$
\begin{eqnarray}
f(X) = b + \sum_i w_i x_i
\end{eqnarray}
$$

In the case of regression, $$f(X)$$ gives us our predicted output, given the input vector $$X$$. In the case of classification, our predicted class is given by 

在回归因子的案例中，如果输入$$X$$函数的话，$$f(X)$$给了我们预测中的输出$$f(X)$$。在分类器的案例中，我们预测中的类来自下面的函数


$$
\begin{eqnarray}
  \mbox{classification} & = & \left\{ \begin{array}{ll}
  	  1 & \mbox{if } f(X) \gt 0 \\
      0 & \mbox{if } f(X) \leq 0
      \end{array} \right.
\tag{1}\end{eqnarray}
$$

Each weight, $$w_i$$, can be interpreted as signifying the relative influence of the input that it's multiplied by, $$x_i$$. The $$b$$ term in the equation is often called the _bias_, because it controls how predisposed the neuron is to firing a 1 or 0, irrespective of the weights. A high bias makes the neuron require a larger input to output a 1, and a lower one makes it easier.

在每个权值中，$$w_i$$，可以被翻译为意味着与输入的相关影响而且可被乘为$$x_i$$。The $$b$$短语在等式中被称为偏差，因为它控制如何让神经元倾向从1变量变为0，并不尊重权值。一个高变量可以让神经元提供一个更高的输入到1的输出，而且一个更小的值会让它变得更简单。

We can get from this formula to a full-fledged neural network by introducing two innovations. The first is the addition of an _activation function_, which turns our linear discriminator into what's called a _neuron_, or a "_unit_" (to dissociate them from the brain analogy). The second innovation is an architecture of neurons which are connected sequentially in _layers_. We will introduce these innovations in that order.

我们可以从这个公式的两个革新中推导出一个成熟的神经网络。第一个是激活函数，
可以把线性识别器变成神经元，或者是一个“单元”，（把它们和人脑类比）第二个革新是可以在层中按次序连接的神经元结构。我们会按这个次序介绍这些革新。

## Activation function

In both artificial and biological neural networks, a neuron does not just output the bare input it receives. Instead, there is one more step, called an _activation function_, analagous to the rate of [action potential](https://en.wikipedia.org/wiki/Action_potential) firing in the brain. The activation function takes the same weighted sum input from before, $$z = b + \sum_i w_i x_i$$, and then transforms it once more before finally outputting it.
同样在人工和生物神经网络中，一个神经元不会只输出它所接收到的信号。所代替的，是还有另外一步，叫做激活函数，你可以把它和大脑中的潜在行为激活.这个激活函数和之前输入的数量相同，$$z = b + \sum_i w_i x_i$$，然后在最终输出前会再次转化。

Many activation functions have been proposed, but for now we will describe two in detail: sigmoid and ReLU. 
许多激活函数都被提出，但是我们目前只能详细说两种函数：sigmoid 和 ReLU。
Historically, the [sigmoid](https://en.wikipedia.org/wiki/Sigmoid_function) function is the oldest and most popular activation function. It is defined as:
历史上，这个sigmoid函数是最老且最为流行的激活函数。它被定义成：

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

$$e$$ denotes the [exponential constant](https://en.wikipedia.org/wiki/E_(mathematical_constant)), roughly equal to 2.71828. A neuron which uses a sigmoid as its activation function is called a _sigmoid neuron_. We first set the variable $$z$$ to our original weighted sum input, and then pass that through the sigmoid function.

$$e$$表示指数常量，(https://en.wikipedia.org/wiki/E_(数学常量))，大概等同于2.71828。一个使用S型作为激励函数的神经元被称为S型神经元。我们首先把 $$z$$变量设置为初始的输入总量，然后让它通过S型函数。

$$
z = b + \sum_i w_i x_i \\
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

At first, this equation may seem complicated and arbitrary, but it actually has a very simple shape, which we can see if we plot the value of $$\sigma(z)$$ as a function of the input $$z$$. 

一开始，这个等式看上去很复杂随意，但其实当我们标出 $$\sigma(z)$$，作为输入 $$z$$后通过函数的输出值后，其实只会出现一个简单的形状。


{:.center}
![sigmoid](/images/sigmoid.png 'sigmoid')


We can see that $$\sigma(z)$$ acts as a sort of "squashing" function, condensing our previously unbounded output to the range 0 to 1. In the center, where $$z = 0$$, $$\sigma(0) = 1/(1+e^{0}) = 1/2$$. For large negative values of $$z$$, the $$e^{-z}$$ term in the denominator grows exponentially, and $$\sigma(z)$$ approaches 0. Conversely, large positive values of $$z$$ shrink $$e^{-z}$$ to 0, so $$\sigma(z)$$ approaches 1.

我们可以看到，$$\sigma(z)$$作为一种“挤压“函数，把我们原先不受限制的输出值浓缩至0到1之间。在这中心， $$z = 0$$, $$\sigma(0) = 1/(1+e^{0}) = 1/2$$.对于$$z$$的负值，在分母上的$$e^{-z}$$函数呈指数式增长，而 $$\sigma(z)$$则会接近零值。
相反的，$$z$$ 的大正值把 $$e^{-z}$$ 收缩为0， 所以$$\sigma(z)$$ 的值会趋近于1.

The sigmoid function is continuously differentiable, and its derivative, conveniently, is $$\sigma^\prime(z) = \sigma(z) (1-\sigma(z))$$. This is important because we have to use calculus to train neural networks, but don't worry about that for now.

S函数是持续变化的可变常量，而它的派生会变为$$\sigma^\prime(z) = \sigma(z) (1-\sigma(z))$$。这很重要因为我们要用微积分来训练神经网络，所以现在不用担心这些。

Sigmoid neurons were the basis of most neural networks for decades, but in recent years, they have fallen out of favor. The reason for this will be explained in more detail later, but in short, they make neural networks that have many layers difficult to train due to the [vanishing gradient problem](https://en.wikipedia.org/wiki/Vanishing_gradient_problem). Instead, most have shifted to using another type of activation function, the _rectified linear unit_, or ReLU for short. Despite its obtuse name, it simply given by $$R(z) = max(0, z)$$.

S神经元在几十年间都是全部神经元的基础，但是在最近的几年中，它们已不常被使用。这其中的细节原因先不综述，简单来说，就是因为“消失斜度问题”，他们会让神经网络有很多层，并难以被训练。

{:.center}
![ReLU](/images/relu.png 'ReLU')

In other words, ReLUs let all positive values pass through unchanged, but just sets any negative value to 0. Although newer activation functions are gaining traction, most deep neural networks these days use ReLU or one of its [closely related variants](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)). 

换句话说，ReLUs让所有的正值都没有改变，但是把所有的负值都设置为0.尽管新的激活函数在获得牵拉力，大多数现在的深度神经网络都在使用ReLU或者与它［联系紧密的变量］。

Regardless of which activation function is used, we can visualize a single neuron with this standard diagram, giving us a nice intuitive visual representation of a neuron's behavior.

无论哪个激励函数被使用，我们都可以用这个标准化图像来视觉化单个神经元，这很好地像我们展示了一个神经元行为的原生视觉呈现。

{:.center}
![ReLU](/images/neuron.png 'Neuron')

The above diagram shows a neuron with three inputs, and outputs a single value $$y$$. As before, we first compute the weighted sum of its inputs, then pass it through an activation function $$\sigma$$.

下列公式显示了有三个输入量的神经元，然后输出一个简单值$$y$$。像从前，我们首先计算输入的总值，然后把它带入激励函数$$\sigma$$。

$$
z = b + w_1 x_1 + w_2 x_2 + w_3 x_3 \\
y = \sigma(z)
$$

You may be wondering what the purpose of an activation function is, and why it is preferred to simply outputting the weighted sum, as we do with the linear classifier from the last chapter. The reason is that a weighted sum is [_linear_](https://en.wikipedia.org/wiki/Linearity) with respect to its inputs, i.e. it has a flat dependence on each of the inputs. In contrast, non-linear activation functions greatly expand our capacity to model curved or otherwise non-trivial functions. This will become clearer in the next section.
你可能会疑问输入激励函数的目的是什么，为什么我们偏好单纯输出这些总量，就像我们对后章中的直线分类器所做的那样。原因是关于输入，这个加权总数是[_线性_](https://en.wikipedia.org/wiki/Linearity)比如，它有一个对每个输入量的绝对依赖。相反，非线性函数可以极大扩大我们建模曲线或其他非平凡函数的容量。在下个章节中，这些会被解释得更清楚。

# Layers

Now that we have described neurons, we can now define neural networks. A neural network is composed of a series of _layers_ of neurons, such that all the neurons in each layer connect to the neurons in the next layer. 
在描述完神经元后，现在我们能够为神经网络下个定义。一个神经网络是由一系列层层排列（layers）的神经元组成的,每一层的神经元都与相邻层级的神经元相关联。{:.center}
![neural network](/images/network.png 'neural network')

Note that when we count the number of layers in a neural network, we only count the layers with connections flowing into them (omitting our first, or _input layer_). So the above figure is of a 2-layer neural network with 1 _hidden layer_. It contains 3 input neurons, 2 neurons in its hidden layer, and 1 output neuron.

注意，当我们在计算一个神经网络中有多少神经层的时候，我们只计算那些有连接流入它们的那些神经层（省略掉第一，或者说是传入层（input layer））。因此，上图中应该有两层神经网络并且其中之一是隐含层。它包含三个传入神经元，两个神经元在它的隐含层中，以及一个传出神经元。

Our computation starts with the input layer on the left, from which we pass the values to the hidden layer, and then in turn, the hidden layer will send its output values to the last layer, which contains our final value.

我们从左侧的传入层开始计算，从这里我们将数值传递给隐含层，然后隐含层再依次将其输出的数值传导给最后一层，从而得到我们最后的数值。

Note that it may look like the three input neurons send out multiple values because each of them are connected to both of the neurons in the hidden layer. But really there is still only one output value per neuron, it just gets copied along each of its output connections. Neurons always output one value, no matter how many subsequent neurons it sends it to.

注意，这三个传入神经元与隐含层的两个神经元相互之间都有连接，因此看起来好像是他们传递了多重数值。但实际上，每个神经元都只输出了一种数值，这个数值只是被复制到每个传出连接中。不论有多少输出对象，神经元们都总是只输出一种数值。

# Forward propagation
# 正向传播

The process of a neural network sending an initial input forward through its layers to the output is called _forward propagation_ or a _forward pass_ and any neural network which works this way is called a _feedforward neural network_. As we shall soon see, there are some neural networks which allow data to flow in circles, but let's not get ahead of ourselves yet... 

一个神经网络将最初的传入数值经过其神经层并最终成为传出数值的过程叫做正向传播（forward propagation or a forward pass）。任何通过类似模式工作的神经网络都可以被称为前馈型神经网络（feed forward neural network）。

Let's demonstrate a forward pass with this interactive demo. Click the 'Next' button in the top-left corner to proceed. You can see a forward pass in action in the following demo. 
让我们通过这个演示案例来展示一个正向传播。按下在左上的“下一个”按钮来继续。你会在之后的样本中看到之后的进程。

{:.center}
![neural network](/images/temp_demo_forward_pass.png 'forward_pass')


# More layers, more expressiveness
# 更多神经层，更强的表达力

Why are hidden layers useful? The reason is that if we have no hidden layers and map directly from inputs to output, each input's contribution on the output is independent of the other inputs. In real-world problems, input variables tend to be highly interdependent and they affect the output in combinatorially intricate ways. The hidden layer neurons allow us to capture subtle interactions among our inputs which affect the final output downstream. 

为什么隐含层是有用的？原因是如果我们没有隐含层，传输路径直接由输入 传导到输出，每个输入之于输出的作用就会完全独立于其他的输入。在现实世界的问题中，输入变量总是倾向于是相互依存的，同时他们对于输出的影响也是结合式的，复杂的。隐含层神经元让我们能够抓住输入信号之间那些微妙的会影响最终输出结果的互动。
Another way to interpret this is that the hidden layers represent higher-level "features" or attributes of our data. Each of the neurons in the hidden layer weigh the inputs differently, learning some different intermediary characteristic of the data, and our output neuron is then a function of these instead of the raw inputs. By including more than one hidden layer, we give the network an opportunity to learn multiple levels of abstraction of the original input data before arriving at a final output. This notion of high-level features will become more concrete in the next chapter when we look closely at the hidden layers.

另一个对于这个问题的解释是，隐含层代表了这些数据更高级别的“特征”或者说属性。每一个隐含层神经元处理输入的方式都不相同，学习数据所内涵的不同特征，然后我们的输出神经元将会成为一个由这些信息整合而成的函数，而不再是原始的输入数据。当有多个隐含层时，这个神经网络就能够在达到最终输出之前，学习到一个输入数据多层次的概念通过下一章对隐含层更深入的了解，高层次的特征这个概念将会更加具体有形。


Recall also that activation functions expand our capacity to capture non-linear relationships between inputs and outputs. By chaining multiple non-linear transformations together through layers, this dramatically increases the flexibility and expressiveness of neural networks. The proof of this is complex and beyond the scope of this book, but it can even be shown that any 2-layer neural network with a non-linear activation function (including sigmoid or ReLU) is a [_universal function approximator_](http://www.sciencedirect.com/science/article/pii/0893608089900208), that is it's theoretically capable of expressing any arbitrary input-to-output mapping. This property is what makes neural networks so powerful.

回顾激活函数对发掘输入与输出之间非线性关系的帮助。通过将多重非线性转换关系通过神经层串联在一起，神经网络的灵活性和表现力得到了显著提升。其证明过程非常复杂，并且超出了本书的讲述范围，但这甚至能够说明任何具有非线性激活函数（包括曲线和正线性单元）的双层神经网络是一个普遍函数近似器（universal function approximator），理论上能够表达任意由输入到输出的映射。而正是这一特质让神经网络如此强大。

# Regression and classification
#衰退和分类
In the demo above, we map a set of inputs to a single output in a forward pass. We can interpret the forward pass as making a prediction about the output, given the input. This is called _regression_ and is one of the primary uses of neural networks. Let's try an example.

在以上的样品中，我们在之后的过程中输入一系列值来获得单个输出值。我们可以给出输出值，再通过对输出值做预测来解读之后的进程。这就叫做_衰退_而且也是神经网络其中基础用法之一。让我们来试一个例子。
-----

[[ THIS IS A DRAFT]]
Random set of dat, 3 cols. 1 regression value

随即设定数据文件，3个关口。1个衰退值

Interactive 3 -> 1
交互3 -> 1


Measure the error. Use L2 error
判定错误。使用L2错误。

Suppose we are trying to model ___...

假定我们在使用模型___..

We have five observed examples of data.

我们有5个观察的数据例子。

Give random weights. Not very accurate.

给出随机值。并不精确

Now a magic trick, I'll give it a new set of weights. Now let's run the examples again, and see that they are all correct now. 

现在一个小把戏，我给予它新的一系列值。我们再次运行这些例子，并且看他们是否现在都正确。

the process of obtaining the correct weights is called training. for now, ignore it, it's a black box. we'll talk about it in the next section.

获取正确值的过程叫做训练。对现在来说，忽视它，这是一个黑盒子。我们可以在下一章中再说这个问题。
-----


What about classification? In the previous chapter, we introduced binary classification by simply thresholding the output at 0; If our output was positive, we'd classify positively, and if it was negative, we'd classify negatively. For neural networks, it would be reasonable to adapt this approach for the final neuron, and classify positively if the output neuron scores above some threshold. For example, we can threshold at 0.5 for sigmoid neurons which are always positive.

分类是什么？在之前的章节中，我们通过单纯介绍了从0开始输出进行的双分类；如果我们的输出值是正的，我们可以正向分类。如果它是负的，我们就要负向分类。对于神经网络来说，为最终的神经元来适应这种方法是合理的，而且而且如果如果输出神经元值超过输入值的话，就可以正向分类。比如，我们可以在总是正值的S型神经元中输入0.5/

But what if we have multiple classes? One option might be to create intervals in the output neuron which correspond to each class, but this would be problematic for reasons that we will learn about when we look at [how neural networks are trained](). Instead, neural networks are adapted for classification by having one output neuron for each class. We do a forward pass and our prediction is the class corresponding to the neuron which received the highest value. Let's have a look at an example.

但如果我们有很多类呢？一个选择可能是在和每类相关的输出神经元中创造间隔，但当我们开始学到我们何时看到 [神经网络如何被训练]()相应的，神经网络通过d每类有一个的输出神经元来适应被分类。我们的进程继续后，我们的预测是获得最高值的神经元。我们来看一个例子。
# Classification of handwritten digits

Let's now tackle a real world example of classification using neural networks, the task of recognizing and labeling images of handwritten digits. We are going to use the [MNIST dataset](http://yann.lecun.com/exdb/mnist/), which contains 60,000 labeled images of handwritten digits sized 28x28 pixels, whose classification accuracy serves as a common benchmark in machine learning research. Below is a random sample of images found in the dataset.

现在让我们来破解一些利用神经网络的分类例子，这些识别和标签手写位数的任务。我们将要使用[MNIST数据集](http://yann.lecun.com/exdb/mnist/)，它有28*28像素的60，000标签图片，他们的分类精确度在机器学习研究中被作为一个普遍的基准，接下来是我在数据集中发现的一个随机的变量。

{:.center}
![MNIST ground truth](/images/temp_fig_mnist_groundtruth.png 'MNIST ground truth')

The way we setup a neural network to classify these images is by having the raw pixel values be our first layer inputs, and having 10 output classes, one for each of our digit classes from 0 to 9. Since they are grayscale images, each pixel has a brightness value between 0 (black) and 255 (white). All the MNIST images are 28x28, so they contain 784 pixels. We can unroll these into a single array of inputs, like in the following figure.

我们要建立起神经网络来分类照片的方法要用生的像素值来成为我们第一分层，并且有10个输出累，每一类小数类的每一个值都从0到9.既然他们有灰度图片，每个像素都有从0（黑色）到255（白色）的明度值。我们可以用简单排列输入值的方法展开这些，就像接下来的图形一样。

{:.center}
![MNIST](/images/temp_fig_mnist.png 'MNIST')

The important thing to realize is that although this network seems a lot more imposing than our simple 3x2x1 network in the previous chapter, it works exactly as before, just with many more neurons. Each neuron in the first hidden layer receives ....
最重要的事是认识到即使在前面的章节中，这个网络看上去像比我们简单的3*2*1的神经网络要更加令人印象深刻，它和之前的一样工作，就像更多的神经元一样。每个在第一隐藏层的神经元都获得了。。

{:.center}
![MNIST demo](/images/temp_demo_mnist_forwardpass.png 'MNIST demo')


# Summary

TBD

# Further reading

 - nielsen
 - kurekenov
 
I