---
layout: chapter
title: "Neural networks"
includes: [mathjax, jquery, convnetjs, dataset, convnet, visualizer]
header_image: "/images/headers/analytical_engine.jpg"
header_text: "“It were much to be desired, that when mathematical processes pass through the human brain instead of through the medium of inanimate mechanism, it were equally a necessity of things that the reasonings connected with operations should hold the same just place as a clear and well-defined branch of the subject of analysis, a fundamental but yet independent ingredient in the science, which they must do in studying the engine.” <a href=\"https://books.google.de/books?id=b8YUDAAAQBAJ&pg=PA16&lpg=PA16\">Sketch of the Analytical Engine (1843), Ada Lovelace</a>"
---
[中文](/ml4a/cn/neural_networks/)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[español](/ml4a/es/neural_networks/)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[français](/ml4a/fr/neural_networks/)

Près d'un siècle avant la conception des réseaux neuronaux, [Ada Lovelace](http://findingada.com/) a décrit l'ambition de construire «[l'analyse du système nerveux]((http://www.thelancet.com/journals/lancet/article/PIIS0140-6736(15)00686-8/fulltext?rss=yes))». Bien que les analogies spéculatives entre les cerveaux et les machines soient aussi anciennes que la philosophie de l'informatique elle-même, ce n'est que quand [Charles Babbage](https://en.wikipedia.org/wiki/Charles_Babbage), le professeur d'Ada, a proposé [Le moteur analytique](https://en.wikipedia.org/wiki/Analytical_Engine) que nous avons imaginés de "calculatrices" ayant des capacités cognitives humaines. Ada n'a pas pour voir son rêve de construire le moteur se concrétiser, car les ingénieurs de l'époque étaient incapables de construire les circuits complexes que nécessitaient ses schémas. Néanmoins, l'idée a été transmise au siècle suivant quand [Alan Turing](https://en.wikipedia.org/wiki/Alan_Turing) l'a cité comme l'inspiration pour le [Jeu d’imitation](http://phil415.pbworks.com/f/TuringComputing.pdf), ce qui allait bientôt être appelé le [«Test de Turing»](https://en.wikipedia.org/wiki/Turing_test). Ses ruminations dans les limites du calcul ont incité le premier boom l'intelligence artificielle, préparant le terrain pour le premier âge d'or des réseaux de neurones.

## La naissance et la renaissance des réseaux de neurones

La récente résurgence des réseaux de neurones est une histoire particulière. Intimement liés aux débuts de l'IA, les réseaux neuronaux ont été formalisés pour la première fois à la fin des années 1940 sous la forme de [machines de type B] (https://en.wikipedia.org/wiki/Unorganized_machine) de Turing, en s'appuyant sur des recherches antérieures sur [plasticité neuronale] (https://en.wikipedia.org/wiki/Hebbian_theory) par des neuroscientifiques et des psychologues cognitifs qui étudient le processus d'apprentissage chez les êtres humains. Alors que la mécanique du développement cérébral était découverte, les informaticiens ont expérimenté des versions idéalisées du potentiel d'action et de la rétropropagation neurale pour simuler ce processus dans des machines.

Aujourd'hui, la plupart des scientifiques déconseillent de prendre cette analogie trop au sérieux, car les réseaux neuronaux sont strictement conçus pour résoudre les problèmes d'apprentissage machine, plutôt que de représenter le cerveau précisément, alors qu'une domaine complètement différent, [les neurosciences computationnelles](https://en.wikipedia.org/wiki/Computational_neuroscience) ont relevé le défi de modéliser fidèlement le cerveau. Néanmoins, la métaphore de l'unité de base des réseaux de neurones le neurone biologique simplifié s'est maintenue au fil des décennies. La progression des neurones biologiques aux neurones artificiels peut être résumée par les figures suivantes.

{% include figure_multi.md path1="/images/neuron-anatomy.jpg"
caption1="Anatomie d'un neurone biologique<br/>Source: <a href=\"https://askabiologist.asu.edu/neuron-anatomy\">ASU school of life sciences</a>" path2="/images/neuron-simple.jpg"
caption2="Corps de neurone simplifié dans un réseau<br/>Source: <a href=\"http://www.generation5.org/content/2000/nn00.asp\">Gurney, 1997. An Introduction to Neural Networks</a>" path3="/images/figures/neuron.png" caption3="Un neurone artificiel<br/>&nbsp;" %}

Les réseaux de neurones ont fait un grand pas en avant lorsque [Frank Rosenblatt] (https://en.wikipedia.org/wiki/Frank_Rosenblatt) a conçu le [Perceptron] (https://en.wikipedia.org/wiki/Perceptron) à la fin des années  1950, un type de classificateur linéaire que nous avons vu dans le [dernier chapitre](/ml4a/machine_learning/). Publiquement financé par la marine américaine, le perceptron Mark 1 a été conçu pour effectuer une reconnaissance d'image à partir d'un ensemble de photocellules, de potentiomètres et de moteurs électriques. Son efficacité à compléter des circuits électriques complexes a conduit le New York Times en 1958 à prédire qu'une machine allait bientôt [«marcher, parler, voir, écrire, se reproduire et être consciente de son existence»](http://query.nytimes.com/gst/abstract.html?res=9D01E4D8173DE53BBC4053DFB1668383649EDE).

Le battage médiatique précoce inspirerait les écrivains de science-fiction pour les décennies à venir, mais l'excitation était beaucoup plus tempérée dans la communauté universitaire. Le livre de 1969 de Marvin Minsky et Seymour Papert, [Perceptrons] (https://en.wikipedia.org/wiki/Perceptrons_ (book)), a démontré divers limitations - [même triviales] (http://users.ecs.soton.ac.uk/harnad/Hypermail/Explaining.Mind96/0140.html) - conduisant par inadvertance à un [déclin d'intérêt] (https://en.wikipedia.org/wiki/AI_winter) à la fois dans le milieu universitaire et dans le grand public, qui avait supposé par erreur que les ordinateurs suivraient simplement le [rythme effréné] (https://en.wikipedia.org/wiki/Moore%27s_law) du pouvoir de calcul. Même Turing lui-même a déclaré que les machines possédaient une intelligence de niveau humain d'ici 2000.

Malgré un certain nombre d'améliorations discrètes mais significatives apportées aux réseaux de neurones dans les années 80 et 90 [[1]] (http://www.iro.umontreal.ca/~vincentp/ift3395/lectures/backprop_old.pdf) [[2]] (http://yann.lecun.org/exdb/publis/pdf/lecun-89e.pdf)[[3]](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf), ils sont restés sur la touche jusqu'aux années 2000. La plupart des applications de l'apprentissage automatique dans les domaines commerciales et industrielles s'est concentré sur les [machines à vecteurs de supports] (https://en.wikipedia.org/wiki/Support_vector_machine) et diverses autres approches. [À partir de 2009] (http://www.cs.utoronto.ca/~gdahl/papers/dbnPhoneRec.pdf) et [surtout à partir de 2012](https://www.technologyreview.com/s/530561/the-revolutionary-technique-that-quietly-changed-machine-vision-forever/), les réseaux neuronaux sont redevenus les algorithmes dominantes de l'apprentissage automatique. Leur résurgence a été largement provoquée par l'émergence des [réseaux de neurones convolutifs] (/ml4a/convnets.html) et de [réseaux de neurones récurrents] (/ml4a/RNNs.html), qui ont dépassés (parfois de manière dramatique) l'état de l'art des méthodes antérieures pour les problèmes clés dans le domaine audiovisuel. De plus, ils ont un certain nombre de nouvelles applications et propriétés inédites qui ont attirés l'attention des artistes et des autres en dehors du domaine de l'IA proprement dit. Ce livre examinera de plus près les réseaux de neurones convolutifs, dans un prochain chapitre dédié.

Bien que de nombreux algorithmes d'apprentissage aient été proposés au fil des années, nous concentrerons surtout notre attention sur les réseaux de neurones pour les raisons suivantes:

  - Ils ont une formulation étonnamment simple et intuitive.
  - Les réseaux neuronaux profonds sont l'état de l'art dans plusieurs problèmes d'apprentissage machine importantes, celles les plus pertinentes pour ce livre.
  - La plupart des utilisations créatives récentes de l'apprentissage automatique ont été faites avec des réseaux de neurones.


## De classificateurs linéaires aux neurones

Rappelez-vous du chapitre précédent que l'entrée d'un classificateur linéaire 2d ou régresseur a la forme:

$$
\begin{eqnarray}
f(x_1, x_2) = b + w_1 x_1 + w_2 x_2
\end{eqnarray}
$$

Plus généralement, dans un nombre quelconque de dimensions, il peut être exprimé comme

$$
\begin{eqnarray}
f(X) = b + \sum_i w_i x_i
\end{eqnarray}
$$

Dans le cas de la régression, $$f(X)$$ nous donne notre sortie prédite, étant donné le vecteur d'entrée $$X$$. Dans le cas de la classification, notre classe prédite est donnée par

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