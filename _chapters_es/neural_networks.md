---
layout: chapter
title: "Neural networks"
includes: [mathjax, jquery, convnetjs, dataset, convnet, visualizer]
header_image: "/images/headers/analytical_engine.jpg"
header_quote: "lovelace"
---

aprendizaje de máquinas = machine learning
neural networks = redes neuronales
machines = máquinas
input = la entrada
output = la salida
bias = sesgo
activation function = función de activación
weighted sum = suma ponderada
sigmoid function = función sigmoide
input neuron = neurona de entrada
output neuron = neurona de salida
forward propagation = propagación hacia delante
feedforward neural network = red neuronal prealimentada

Casi un siglo antes de que las redes neuronales fueran primero concebidas, [Ada Lovelace](http://findingada.com/) describió una ambición por construir un "[cálculo del sistema nervioso(http://www.thelancet.com/journals/lancet/article/PIIS0140-6736(15)00686-8/fulltext?rss=yes)." Aunque analogías especulativas sobre mentes y máquinas , no fue hasta que el profesor de Ada, Charles Baggage, propuso la Máquina analítica que empezamos a concebir de las "calculadoras" teniendo capacidades cognitivas humanas. Ada no viviría para ver realizado su sueño de contruir una máquina similar a la que propuso Baggage, ya que lo ingenieros de su época eran incapaces de producir los circuitos complejos que sus esquemas requerían. Sin embargo, la idea sobrevivió hasta el siguiente siglo cuando Alan Turing la citó como inspiración para el Juego de Imitación. Sus reflexiones sobre los límites de la computación incitaron el primer auge en inteligencia artificial, la cual abrió paso para la primera época dorada de las redes neurales.  

Nearly a century before neural networks were first conceived, [Ada Lovelace](http://findingada.com/) described an ambition to build a "[calculus of the nervous system](http://www.thelancet.com/journals/lancet/article/PIIS0140-6736(15)00686-8/fulltext?rss=yes)." Although speculative analogies between brains and machines are as old as the philosophy of computation itself, it wasn't until Ada's teacher [Charles Babbage](https://en.wikipedia.org/wiki/Charles_Babbage) proposed the [Analytical engine](https://en.wikipedia.org/wiki/Analytical_Engine) that we conceived of "calculators" having humanlike cognitive capacities.
Ada would not live to see her dream of building the engine come to fruition, as engineers of the time were unable to produce the complex circuitry her schematics required.
Nevertheless, the idea was passed on to the next century when [Alan Turing](https://en.wikipedia.org/wiki/Alan_Turing) cited it as the inspiration for the [Imitation Game](http://phil415.pbworks.com/f/TuringComputing.pdf), what soon came to be called the "[Turing Test](https://en.wikipedia.org/wiki/Turing_test)." His ruminations into the extreme limits of computation incited the first boom of artificial intelligence, setting the stage for the first golden age of neural networks.

## The once and future king

## El rey del presente y del futuro

El resurgimiento reciente de las redes neuronales es una historia peculiar. Íntimamente conectadas a los primeros días de la inteligencia artificial, las redes neuronales se formalizaron a finales de los años 40s en la forma de máquinas tipo-B de Turing y basadas en investigaciones de plasticidad neuronal conducidas por neurocientíficos y psicólogos cognitivos que estudiaban el proceso de aprendizaje en los seres humanos. A medida que se descubrió cómo se desarrolla el cerebro, los científicos de la computación comenzaron a experimentar con versiones idealizadas de acción potencial y retropropagación neural para simular el proceso en máquinas.

The recent resurgence of neural networks is a peculiar story. Intimately connected to the early days of AI, neural networks were first formalized in the late 1940s in the form of Turing's [B-type machines](https://en.wikipedia.org/wiki/Unorganized_machine), drawing upon earlier research into [neural plasticity](https://en.wikipedia.org/wiki/Hebbian_theory) by neuroscientists and cognitive psychologists studying the learning process in human beings. As the mechanics of brain development were being discovered, computer scientists experimented with idealized versions of action potential and neural backpropagation to simulate the process in machines.

Hoy en día, la mayoría de los científicos nos advierten que deberíamos tener cuidado con esta analogía, ya que las redes neuronales fueron diseñadas para resolver problemas de "machine learning" y no para representar el cerebro con precisión. Sin embargo, la idea de que una neurona biológica simplificada representa la unidad central de una red neuronal es una metáfora que ha perdurado a través de las décadas. La progresión de las neuronas biológicas a las neuronas artificiales se puede resumir con la siguiente gráfica.

Today, most scientists caution against taking this analogy too seriously, as neural networks are strictly designed for solving machine learning problems, rather than accurately depicting the brain. Nevertheless, the metaphor of the core unit of neural networks as a simplified biological neuron has stuck over the decades. The progression from biological neurons to artificial ones can be summarized by the following figures.

{% include neurons.html %}

Las redes neuronales dieron un gran paso adelante cuando Frank Rosenblatt diseño el Perceptron, un tipo de clasificador linear que cubrimos en el capítulo anterior, a finales de los años cincuenta. Financiado públicamente por la Armada de los Estados Unidos, el perceptrón Mark 1 fue diseñado para reconocer imágenes a partir de una serie de fotocélulas, potenciómetros y motores eléctricos. Fue tan efectivo completando circuitos eléctricos complejos que en 1958 el periódico New York Times predijo que una máquina pronto podría "caminar, hablar, ver, escribir, reproducirse y ser consciente de su propia existencia".

Neural networks took a big step forward when [Frank Rosenblatt](https://en.wikipedia.org/wiki/Frank_Rosenblatt) devised the [Perceptron](https://en.wikipedia.org/wiki/Perceptron) in the late 1950s, a type of linear classifier that we saw in the [last chapter](/ml4a/machine_learning/). Publicly funded by the U.S. Navy, the Mark 1 perceptron was designed to perform image recognition from an array of photocells, potentiometers, and electrical motors. Its effectiveness at completing complex electrical circuits lead the New York Times in 1958 to predict that a machine would soon ["walk, talk, see, write, reproduce itself and be conscious of its existence"](http://query.nytimes.com/gst/abstract.html?res=9D01E4D8173DE53BBC4053DFB1668383649EDE).

Este tipo de logros inspiró por décadas a escritores de ciencia ficción, aun cuando la emoción dentro de la comunidad académica fue mucho más reservada. En 1969 el libro de Marvin Minsky y Seymour Papert, Perceptrons, demostró varias limitaciones de las redes neuronales, lo cual condujo inadvertidamente a una disminución de interés por parte del mundo académico y el público general. Ambos habían asumido erróneamente que las computadoras seguirían avanzando al mismo ritmo vertiginoso que el poder computacional. Incluso Turing una vez dijo que las máquinas poseerían una inteligencia comparable a la humana para el año 2000 - el mismo año que tuvimos el susto Y2K.

The early hype would inspire science fiction writers for decades to come, but the excitement was far more tempered in the academic community. Marvin Minsky's and Seymour Papert's 1969 book, [Perceptrons](https://en.wikipedia.org/wiki/Perceptrons_(book)), demonstrated various-—[even trivial](http://users.ecs.soton.ac.uk/harnad/Hypermail/Explaining.Mind96/0140.html)—limitations, inadvertently leading to a [decline of interest](https://en.wikipedia.org/wiki/AI_winter) within both academia and the general public, who had mistakenly assumed computers would simply keep up with the [breakneck pace](https://en.wikipedia.org/wiki/Moore%27s_law) of computational power. Even Turing himself said machines would possess human-level intelligence by the year 2000 -- the year we had the [Y2K scare](https://en.wikipedia.org/wiki/Year_2000_problem).

Despite a number of quiet but significant improvements to neural networks in the 80s and 90s [[1]](http://www.iro.umontreal.ca/~vincentp/ift3395/lectures/backprop_old.pdf)[[2]](http://yann.lecun.org/exdb/publis/pdf/lecun-89e.pdf)[[3]](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf), they remained on the sidelines through the 2000s, with most commercial and industrial applications of machine learning favoring [support vector machines](https://en.wikipedia.org/wiki/Support_vector_machine) and various other approaches. [Starting in 2009](http://www.cs.utoronto.ca/~gdahl/papers/dbnPhoneRec.pdf) and [especially ramping up from 2012](https://www.technologyreview.com/s/530561/the-revolutionary-technique-that-quietly-changed-machine-vision-forever/), neural networks have once again become the dominant strain of ML algorithms. Their resurgence was largely brought about by the emergence of [convolutional](/ml4a/convnets.html) and [recurrent neural networks](/ml4a/RNNs.html), which have surpassed (sometimes dramatically so) previous state-of-the-art methods for key problems in the audiovisual domain. But more interestingly, they have a number of new applications and properties not seen before, especially of a kind that has piqued the interest of artists and others from outside the AI field proper. This book will look more closely at convolutional neural networks in particular several chapters from now.

Aunque se han propuesto muchos algoritmos de aprendizaje automático a lo largo de los años, nos enfocaremos en la redes neuronales ya que:

- Tienen una formulación sorprendentemente simple e intuitiva.
- Las redes neuronales de aprendizaje profundo (en inglés, deep neural networks) constituyen la tecnología de vanguardia para varias de las tareas que discutiremos en este libro.
- La mayoría de las aplicaciones creativas del aprendizaje de máquinas se han hecho con redes neuronales.

Although many learning algorithms have been proposed over the years, we will mostly focus our attention on neural networks because:

 - They have a surprisingly simple and intuitive formulation.
 - Deep neural networks are the current state-of-the-art in several important machine learning tasks, the ones most relevant to this book.
 - Most of the recent creative uses of machine learning have been made with neural networks.


## De clasificadores lineales a neuronas

## From linear classifiers to neurons

Recuerda del capítulo anterior que la entrada a un regressor o un clasificador lineal en 2d tiene la forma:

Recall from the previous chapter that the input to a 2d linear classifier or regressor has the form:

$$
\begin{eqnarray}
f(x_1, x_2) = b + w_1 x_1 + w_2 x_2
\end{eqnarray}
$$

Si lo extendemos a cualquier número de dimensions, se puede expresar como:

More generally, in any number of dimensions, it can be expressed as

$$
\begin{eqnarray}
f(X) = b + \sum_i w_i x_i
\end{eqnarray}
$$

En el caso de la regresión, $$f(X)$$ nos da el resultado predicho, dado el vector de entrada $$X$$. En el caso de la clasificación, la clase predicha está dada por:

In the case of regression, $$f(X)$$ gives us our predicted output, given the input vector $$X$$. In the case of classification, our predicted class is given by

$$
\begin{eqnarray}
  \mbox{classification} & = & \left\{ \begin{array}{ll}
  	  1 & \mbox{if } f(X) \gt 0 \\
      0 & \mbox{if } f(X) \leq 0
      \end{array} \right.
\tag{1}\end{eqnarray}
$$

Podemos interpretar que cada peso, $$w_i$$, representa la influencia relativa de la entrada por la cual se multiplica, $$x_i$$. A menudo al término $$b$$ se le llama sesgo (en inglés bias), ya que controla que tan predispuesta está la neurona a disparar un 1 o un 0 independiente de los pesos. Un sesgo alto hace que la neurona requira una entrada más alta para generar una salida de 1. Un sesgo bajo lo hace más fácil. 

Each weight, $$w_i$$, can be interpreted as signifying the relative influence of the input that it's multiplied by, $$x_i$$. The $$b$$ term in the equation is often called the _bias_, because it controls how predisposed the neuron is to firing a 1 or 0, irrespective of the weights. A high bias makes the neuron require a larger input to output a 1, and a lower one makes it easier.

Podemos obtener una verdadera red neuronal a partir de esta fórmula si introducimos dos inovaciones. La primera es la adición de una función de activación, la cual transforma nuestra discirimador lineal en lo que se llama una _neurona_, o "_unidad_" (para disociarlo de analogías del cerebro). La segunda inovación consiste organizar las neuronas de una manera particular, una arquitectura de neuronas conectadas secuencialmente en _capas_. Cubriremos estas dos inovaciones en orden.

We can get from this formula to a full-fledged neural network by introducing two innovations. The first is the addition of an _activation function_, which turns our linear discriminator into what's called a _neuron_, or a "_unit_" (to dissociate them from the brain analogy). The second innovation is an architecture of neurons which are connected sequentially in _layers_. We will introduce these innovations in that order.

## Función de activación

## Activation function

Tanto en las redes neuronales artificiales como biológicas, una neurona no sólo transmite la entrada que recibe. Existe un paso adicional, una _función de activación_, que es análoga a la tasa de potencial de acción disparando en el cerebro. La función de activación utiliza la misma suma ponderada de la entrada anterior, $$z = b + \sum_i w_i x_i$$, y la transforma una vez más como salida.

In both artificial and biological neural networks, a neuron does not just output the bare input it receives. Instead, there is one more step, called an _activation function_, analagous to the rate of [action potential](https://en.wikipedia.org/wiki/Action_potential) firing in the brain. The activation function takes the same weighted sum input from before, $$z = b + \sum_i w_i x_i$$, and then transforms it once more before finally outputting it.

Se han propuesto muchas funciones de activación, pero por ahora describiremos solamente dos en detalle: la sigmoide y ReLU.

Many activation functions have been proposed, but for now we will describe two in detail: sigmoid and ReLU.

Históricamente, la función sigmoide es la función de activación mas antiguia y popular. Se define como:

Historically, the [sigmoid](https://en.wikipedia.org/wiki/Sigmoid_function) function is the oldest and most popular activation function. It is defined as:

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

$$e$$ denota la constante exponencial, que es aproximadamente igual a 2,71828. Una neurona que utiliza la sigmóide como función de activación se le llama _neurona sigmoide_. Primero establecemos que la variable $$z$$ equivale a nuestra suma ponderada de entrada y después la pasamos a través de la función sigmóide. 

$$e$$ denotes the [exponential constant](https://en.wikipedia.org/wiki/E_(mathematical_constant)), roughly equal to 2.71828. A neuron which uses a sigmoid as its activation function is called a _sigmoid neuron_. We first set the variable $$z$$ to our original weighted sum input, and then pass that through the sigmoid function.

$$
z = b + \sum_i w_i x_i \\
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

Aunque la ecuación parece complicada y arbitraria, en realidad tiene una forma bastante simple. La podemos ver si trazamos el valor de $$\sigma(z)$$ como función de la entrada $$z$$.

At first, this equation may seem complicated and arbitrary, but it actually has a very simple shape, which we can see if we plot the value of $$\sigma(z)$$ as a function of the input $$z$$.

{% include figure.html path="/images/figures/sigmoid.png" caption="Sigmoid activation function" %}

Podemos ver que $$\sigma(z)$$ actúa como una especie de función "aplastadora", comprimiendo nuestra salida a un rango de 0 a 1. En el centro, donde $$z = 0$$, $$\sigma(0) = 1/(1+e^{0}) = 1/2$$. Para valores negativos grandes de $$z$$, el término $$e^{-z}$$ en el denominador crece exponencialmente, y $$\sigma(z)$$ se aproxima a 0. Al contrario, valores positivos grandes de $$z$$ reducen $$e^{-z}$$ hacia 0, y $$\sigma(z)$$ se aproxima a 1. 

We can see that $$\sigma(z)$$ acts as a sort of "squashing" function, condensing our previously unbounded output to the range 0 to 1. In the center, where $$z = 0$$, $$\sigma(0) = 1/(1+e^{0}) = 1/2$$. For large negative values of $$z$$, the $$e^{-z}$$ term in the denominator grows exponentially, and $$\sigma(z)$$ approaches 0. Conversely, large positive values of $$z$$ shrink $$e^{-z}$$ to 0, so $$\sigma(z)$$ approaches 1.

La función sigmoide es continuamente diferenciable, y su derivada convenientemente es $$\sigma^\prime(z) = \sigma(z) (1-\sigma(z))$$. Este detalle indica que tenemos que usar cálculo para entrenar una red neuronal - pero no nos preocuparemos por eso en este capítulo. 

The sigmoid function is continuously differentiable, and its derivative, conveniently, is $$\sigma^\prime(z) = \sigma(z) (1-\sigma(z))$$. This is important because we have to use calculus to train neural networks, but don't worry about that for now.

Las funciones sigmóides fueron la base de la mayoría de las redes neuronales por muchas décadas, aunque en años recientes han perdido popularidad. Explicaremos la razón en detalle en los próximos capítulos, pero la versión corta es que las redes neuronales de muchas capas se vuelven muy difíciles de entrenar dado el problema de desaparición de gradiente. En su lugar, la mayoría de las redes neuronales actuales usan otro tipo de función de activación llamada "rectified linear unit" o ReLU. A pesar del nombre complicado, se define simplemente como $$R(z) = max(0, z)$$.

Sigmoid neurons were the basis of most neural networks for decades, but in recent years, they have fallen out of favor. The reason for this will be explained in more detail later, but in short, they make neural networks that have many layers difficult to train due to the [vanishing gradient problem](https://en.wikipedia.org/wiki/Vanishing_gradient_problem). Instead, most have shifted to using another type of activation function, the _rectified linear unit_, or ReLU for short. Despite its obtuse name, it is simply defined as $$R(z) = max(0, z)$$.

{% include figure.html path="/images/figures/relu.png" caption="ReLU activation function" %}

En otras palabras, las ReLUs permiten el paso de todos los valores positivos sin cambiarlos, pero asigna todos los valores negativos a 0. Aunque existen funciones de activación aún más recientes, la mayoría de las redes neuronales de hoy utilizan ReLU o una de sus variantes.

In other words, ReLUs let all positive values pass through unchanged, but just sets any negative value to 0. Although newer activation functions are gaining traction, most deep neural networks these days use ReLU or one of its [closely related variants](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)).

Independiente de la función de activación que utilizemos, podemos visualizar una neurona individual con el siguiente diagrama, una visual representativa e intuitiva del comportamiento de una neurona. 

Regardless of which activation function is used, we can visualize a single neuron with this standard diagram, giving us a nice intuitive visual representation of a neuron's behavior.

{% include figure.html path="/images/figures/neuron.png" caption="An artificial neuron" %}

Este diagrama muestra una neurona con tres entradas, que genera un único valor como salida. Como en el caso anterior, primero calculamos la suma ponderada de sus entradas, y después pasamos la suma a través de una función de activacion sigmoide.


The above diagram shows a neuron with three inputs, and outputs a single value $$y$$. As before, we first compute the weighted sum of its inputs, then pass it through an activation function $$\sigma$$.

$$
z = b + w_1 x_1 + w_2 x_2 + w_3 x_3 \\
y = \sigma(z)
$$

Quizás te estás preguntando cuál es el propósito de una función de activación, y por qué preferimos usarla en vez de la suma ponderada - como lo hacemos con el clasificador lineal del capítulo anterior. La razón es que la suma ponderada es lineal con respecto a sus entradas. En cambio, las funciones de activación no-lineales nos ayudan a modelar funciones curvas o no triviales. Esto quedará más claro en la siguiente sección. 

You may be wondering what the purpose of an activation function is, and why it is preferred to simply outputting the weighted sum, as we do with the linear classifier from the last chapter. The reason is that the weighted sum, $$z$$, is [_linear_](https://en.wikipedia.org/wiki/Linearity) with respect to its inputs, i.e. it has a flat dependence on each of the inputs. In contrast, non-linear activation functions greatly expand our capacity to model curved or otherwise non-trivial functions. This will become clearer in the next section.

# Capas

# Layers

Ahora que hemos descrito una neurona, podemos definir una red neuronal. Una red neuronal consiste en una serie de _capas_ de neuronas. Específicamente,  todas las neuronas de una capa se conectan a las neuronas de la siguiente capa. 

Now that we have described neurons, we can now define neural networks. A neural network is composed of a series of _layers_ of neurons, such that all the neurons in each layer connect to the neurons in the next layer.

{% include figure.html path="/images/figures/neural-net.png" caption="A 2-layer neural network" %}

Un detalle es que cuanto contamos el número de capas en una red neuronal, sólo contamos las capas con entradas (omitimos la primera _capa de entrada_). La figura anterior representa una red neuronal de 2 capas con 1 capa oculta. Contiene 3 neuronas de entrada, 2 neuronas en la capa oculta, y 1 neurona de salida. 

Note that when we count the number of layers in a neural network, we only count the layers with connections flowing into them (omitting our first, or _input layer_). So the above figure is of a 2-layer neural network with 1 _hidden layer_. It contains 3 input neurons, 2 neurons in its hidden layer, and 1 output neuron.

Nuestro cálculo comienza con la capa de entrada a la izquierda, de la cual pasamos valores a la capa oculta. De ahí, la capa oculta envía valores de salida a la última capa, que contiene el valor final. 

Our computation starts with the input layer on the left, from which we pass the values to the hidden layer, and then in turn, the hidden layer will send its output values to the last layer, which contains our final value.

Aunque pareciera que cada una de las tres neuronas de entrada envía múltiples valores de salida a la capa oculta, en realidad solamente hay un valor de salida por neurona. Las neuronas siempre producen 1 valor, independientemente de cuántas conexiones de salida tengan. 


Note that it may look like the three input neurons send out multiple values because each of them are connected to both of the neurons in the hidden layer. But really there is still only one output value per neuron, it just gets copied along each of its output connections. Neurons always output one value, no matter how many subsequent neurons it sends it to.

# Regresión

# Regression

Llamamos _propagación hacia delante_ (en inglés, forward propagation o forward pass) al proceso por la cual una red neuronal envía su entrada a través de sus capas hacia la salida. A las redes neuronales que funcionan de esta manera se les llama _red neuronal prealimentada_ (en inglés, feedforward neural network). Ya pronto veremos que algunas redes neuronales permiten que los datos fluyan en círculos. 

The process of a neural network sending an initial input forward through its layers to the output is called _forward propagation_ or a _forward pass_ and any neural network which works this way is called a _feedforward neural network_. As we shall soon see, there are some neural networks which allow data to flow in circles, but let's not get ahead of ourselves yet...

Por ahora demostraremos una propagación hacia delante con este ejemplo interactivo. Dale click al botón 'Siguiente' en la esquina superior derecha para continuar. 

Let's demonstrate a forward pass with this interactive demo. Click the 'Next' button in the top-right corner to proceed.

{% include demo_insert.html path="/demos/simple_forward_pass/" parent_div="post" %}

# Más capas, más potencial de expresión

# More layers, more expressiveness

¿Por qué son tan útiles las capas ocultas? La razón es que si no tuvieramos capas ocultas y tuvieramos que trazar una conexión directa entre nuestras entradas y nuestra salida - la contribución de cada entrada hacia el valor de salida sería independiente de las otras entrdas. En la mayoría de los problemas del mundo real, las variables de entrada tienden a ser altamente interdependientes y afectan la salida de forma combinatoria y compleja. Las neuronas de las capas ocultas nos permiten capturar interacciones sutiles entre nuestras entradas.

Why are hidden layers useful? The reason is that if we have no hidden layers and map directly from inputs to output, each input's contribution on the output is independent of the other inputs. In real-world problems, input variables tend to be highly interdependent and they affect the output in combinatorially intricate ways. The hidden layer neurons allow us to capture subtle interactions among our inputs which affect the final output downstream.

Otra manera de interpretar esta idea es que las capas ocultas representan "características" a nivel superior o atributos de nuestros datos. Cada una de las neuronas de una capa oculta sopesa sus entradas de forma diferente, y de esta manera aprende características diferentes de los datos. Nuestra neurona de salida logra capturar estas características intermediarias, no sólo las entradas originales. Al incluir más de una capa oculta, permitimos que la red neuronal pueda aprender sobre varios niveles de abstracción de los datos. En el próximo capítulo aprenderemos más sobre las capas ocultas y sobr esta noción de "características" de alto nivel.

Another way to interpret this is that the hidden layers represent higher-level "features" or attributes of our data. Each of the neurons in the hidden layer weigh the inputs differently, learning some different intermediary characteristic of the data, and our output neuron is then a function of these instead of the raw inputs. By including more than one hidden layer, we give the network an opportunity to learn multiple levels of abstraction of the original input data before arriving at a final output. This notion of high-level features will become more concrete [in the next chapter when we look closely at the hidden layers](/ml4a/looking_inside_neural_nets/).

Recuerda también que las funciones de activación también pueden apliar nuestra capacidad 

Recall also that activation functions expand our capacity to capture non-linear relationships between inputs and outputs. By chaining multiple non-linear transformations together through layers, this dramatically increases the flexibility and expressiveness of neural networks. The proof of this is complex and beyond the scope of this book, but it can even be shown that any 2-layer neural network with a non-linear activation function (including sigmoid or ReLU) and enough hidden units is a [_universal function approximator_](http://www.sciencedirect.com/science/article/pii/0893608089900208), that is it's theoretically capable of expressing any arbitrary input-to-output mapping. This property is what makes neural networks so powerful.

# Clasificación

# Classification

What about classification? In the previous chapter, we introduced binary classification by simply thresholding the output at 0; If our output was positive, we'd classify positively, and if it was negative, we'd classify negatively. For neural networks, it would be reasonable to adapt this approach for the final neuron, and classify positively if the output neuron scores above some threshold. For example, we can threshold at 0.5 for sigmoid neurons which are always positive.

But what if we have multiple classes? One option might be to create intervals in the output neuron which correspond to each class, but this would be problematic for reasons that we will learn about when we look at [how neural networks are trained](/ml4a/how_neural_networks_are_trained/). Instead, neural networks are adapted for classification by having one output neuron for each class. We do a forward pass and our prediction is the class corresponding to the neuron which received the highest value. Let's have a look at an example.

# Classification of handwritten digits

Let's now tackle a real world example of classification using neural networks, the task of recognizing and labeling images of handwritten digits. We are going to use the [MNIST dataset](http://yann.lecun.com/exdb/mnist/), which contains 60,000 labeled images of handwritten digits sized 28x28 pixels, whose classification accuracy serves as a common benchmark in machine learning research. Below is a random sample of images found in the dataset.

{% include figure.html path="/images/figures/fig_mnist_groundtruth.png" caption="A random sample of MNIST handwritten digits" %}

The way we setup a neural network to classify these images is by having the raw pixel values be our first layer inputs, and having 10 output classes, one for each of our digit classes from 0 to 9. Since they are grayscale images, each pixel has a brightness value between 0 (black) and 255 (white). All the MNIST images are 28x28, so they contain 784 pixels. We can unroll these into a single array of inputs, like in the following figure.

{% include figure.html path="/images/figures/mnist-input.png" caption="How to input an image into a neural network" %}

The important thing to realize is that although this network seems a lot more imposing than our simple 3x2x1 network in the previous chapter, it works exactly as before, just with many more neurons. Each neuron in the first hidden layer receives all the inputs from the first layer. For the output layer, we'll now have _ten_ neurons rather than just one, with full connections between it and the hidden layer, as before. Each of the ten output neurons is assigned to one class label; the first one is for  the digit `0`, the second for `1`, and so on.

After the neural network has been trained -- something we'll talk about in more detail [in a future chapter](/ml4a/how_neural_networks_are_trained/) -- we can predict the digit associated with unknown samples by running them through the same network and observing the output values. The predicted digit is that whose output neuron has the highest value at the end. The following demo shows this in action; click "next" to flip through more predictions.

{% include demo_insert.html path="/demos/forward_pass_mnist/" parent_div="post" %}

# Recursos adicionales

# Further reading
