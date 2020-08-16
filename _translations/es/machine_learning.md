---
layout: chapter
title: "Aprendizaje automático"
header_image_hide_: "/images/headers/mnist_zeros.png"
header_text_hide: "_"
---


Los métodos de aprendizaje supervisado incluyen, entre otros, redes neuronales, que serán la categoría principal de métodos que se incluya por ahora. Son neutrales en cuanto a la interpretación.

En el sentido más amplio, los métodos de aprendizaje supervisado asignan información estructurada a información estructurada. Al hacerlo, forman también una representación de la cosa en sí misma.
---

Ya has escuchado que el aprendizaje automático se refiere a un amplio conjunto de técnicas que permiten a las computadoras aprender de los datos. ¿Pero aprender exactamente qué, y cómo? Consideremos varios ejemplos concretos en los que se pueden aplicar técnicas de aprendizaje automático.

**Ejemplo 1**: Supongamos que eres un climatólogo que está tratando de diseñar un programa de computadora que pueda predecir si lloverá o no un día determinado. ¡Resulta que esto es difícil! Pero intuitivamente entendemos que la lluvia tiene algo que ver con la temperatura, la presión atmosférica, la humedad, el viento, la capa de nubes, la ubicación y la época del año, etc.

**Ejemplo 2**: Gmail, Yahoo, y otros servicios de correo electrónico proporcionan herramientas para filtrar automáticamente los correos no deseados antes de que lleguen a su bandeja de entrada. Como en el ejemplo de la lluvia, tenemos algunas intuiciones sobre esta tarea. Los correos electrónicos que contienen frases como "haga $$$ ahora" o "píldoras de pérdida de peso gratis" probablemente sean sospechosos, y seguramente podemos pensar en algunas más. Por supuesto, la presencia de un término sospechoso no garantiza que sea spam y, por lo tanto, no podemos adoptar el enfoque ingenuo de etiquetar como correo no deseado cualquiera que contenga una frase sospechosa.

La forma en que un programador tradicional podría resolver estos problemas es diseñar cuidadosamente una serie de reglas o declaraciones condicionales que se prueban durante la ejecución para determinar el resultado. En el ejemplo del spam, esto podría tomar la forma de un [árbol de decisión](___): al recibir un correo electrónico, verifique si es de un remitente desconocido; si lo es, compruebe si la frase "¡perder peso ahora!" aparece, y si aparece y hay un link a un sitio web desconocido, clasifíquelo como spam. Nuestro árbol de decisión sería mucho más grande y más complicado que esto, pero se caracterizaría por una secuencia de declaraciones si-entonces que llevan a una decisión. 

Dicha estrategia, comúnmente llamada "basada en reglas" o "[sistema experto](https://es.wikipedia.org/wiki/Sistema_experto)", sufre de dos debilidades principales. Primero, requiere gran cantidad de orientación experta e ingeniería manual que puede llevar mucho tiempo y ser costosa. Además, las palabras de activación de spam y los patrones climáticos globales cambian continuamente, y tendríamos que reprogramarlos de vez en cuando para que sigan siendo efectivos. En segundo lugar, un enfoque basado en reglas no generaliza. Ten en cuenta que nuestro árbol de decisiones de spam no se adaptará a la predicción de la lluvia, o viceversa, ni se aplicará fácilmente a otros problemas de los que no hemos hablado. Los sistemas expertos como estos son de dominio específico y si nuestra tarea cambia, aunque sea un poco, nuestro algoritmo cuidadosamente diseñado debe reconstruirse desde cero.

{:.section}
Aprendiendo de observaciones pasadas

Con el aprendizaje automático adoptamos otro enfoque. Comenzamos reduciendo los dos ejemplos diferentes explicados antes a, esencialmente, la misma tarea genérica: dado un conjunto de observaciones sobre algo, tome una decisión o _ ** clasificación ** _. Lluvia o no lluvia, spam o no spam. En otros tipos de problemas podemos tener más de dos opciones, o podemos tener un valor continuo para predecir, por ejemplo: cuánto lloverá. En este último caso, llamamos a este problema _**regresión**_. 

En nuestros dos ejemplos hemos planteado un único problema abstracto: determinar la relación entre nuestras observaciones o datos y nuestra tarea deseada. Esto puede tomar la forma de una función o modelo que toma nuestras observaciones y calcula una decisión partiendo de ellas. El modelo se determina a partir de la experiencia, dándole un conjunto de pares de observaciones y decisiones conocidas. Una vez que tenemos el modelo, podemos hacer predicciones de resultados. [todo esto es descuidado, arregla esto]

[Observaciones conocidas] -> [Aprendizaje] <- [Resultados conocidos]
||
[Observaciones desconocidas] -> [ Modelo ] ->  [Resultados predichos]


El aprendizaje automático también propone que una relación funcional de este tipo puede ser _aprendida_ de las observaciones pasadas y sus resultados conocidos. Para el problema de predicción de lluvia, podemos tener una base de datos con miles de ejemplos en los que se midieron las variables que consideramos importantes (presión, temperatura, etc.) y sabemos si realmente llovió o no esos días. En el ejemplo de spam, podemos tener una base de datos de correos electrónicos que fueron etiquetados como spam o no spam por un humano. Usando estos datos, podemos crear una función que sea capaz de modificar su propia estructura interna en respuesta a nuevas observaciones, con el fin de poder mejorar su capacidad para realizar la tarea con precisión.Formalmente, el conjunto de los ejemplos anteriores con sus resultados conocidos suele llamarse una _verdad fundamental_ (_ground truth_) y se usa como _conjunto de entrenamiento_ para entrenar nuestro algoritmo predictivo. [[ todo esto necesita ser arreglado/fusionado con la sección anterior ]]

En términos más generales, lo que se ha definido en esta sección se llama _**aprendizaje supervisado**_ y es una de las ramas fundamentales del aprendizaje automático. _**Aprendizaje no supervisado**_ se refiere a

More generally, what\'s been defined in this section is called _**supervised learning**_ and is one of the foundational branches of machine learning. _**Unsupervised learning**_ refers to tareas que involucran datos que no están etiquetados, y _**aprendizaje por refuerzo**_ (_**reinforcement learning**_) es un híbrido de los dos, pero los veremos más adelante.

¿basado-en-datos?

{:.section}
El algoritmo de aprendizaje automático más simple: un clasificador lineal

Hemos introducido la noción de un algoritmo que hace una serie de observaciones empíricas sobre algo y las utiliza para tomar una decisión sobre algo.

Ahora haremos nuestro primer modelo predictivo, un simple _clasificador lineal_. Un clasificador lineal se define como una función de nuestros datos, $$X$$, y 

Tomemos nuestro primer ejemplo, el de predecir si lloverá o no un día determinado. Utilizaremos un conjunto de datos simplificado que consta de solo dos observaciones: presión atmosférica y humedad. Supongamos que tenemos un conjunto de datos con 6 días de datos pasados.

* notas
- haz dos columnas, pon humedad + presión en ellas 

|**Humedad (%)**|**Presión (kPa)**|**¿Lluvia?**|
|==|==|==|
|29|101.7|-|
|60|98.6|+|
|40|101.1|-|
|62|99.9|+|
|39|103.2|-|
|51|97.6|+|
|46|102.1|-|
|55|100.2|+|

Pongamos esto en un gráfico 2d.

{:.center}
![linear classifier](/images/lin_classifier_2d.png 'linear classifier')

Intuitivamente, vemos que los días lluviosos tienden a tener baja presión y alta humedad, y los días no lluviosos son lo opuesto. Si miramos el gráfico, vemos que podemos separar fácilmente las dos clases con una línea.

Si dejamos que $$x_1$$ represente la humedad, y $$x_2$$ represente la presión, podemos trazar una línea en nuestro gráfico con la siguiente ecuación:

$$w_1*x_1 + w_2*x_2 + b = 0$$

donde $$w_1$$, $$w_2$$, y $$b$$ son coeficientes que podemos elegir libremente. Si establecemos $$w_1 = 5$$, $$w_2 = 6$$, y $$b = 1.2$$, y luego trazamos la línea resultante en el gráfico, vemos que separa perfectamente nuestras dos clases. Llamamos a esta línea nuestro _límite de decisión_.

(la ecuación debe escribirse al lado de la línea, Ax+By+C)

{:.center}
![linear classifier](/images/lin_classifier_2d.png 'linear classifier')

Supongamos que conocemos la humedad y la presión de hoy, y se nos pide predecir si lloverá o no. Digamos que tenemos $$x_1 = 20$$ y $$x_2 = 3$$. Podemos trazar este nuevo punto en nuestro gráfico.

(ahora con un nuevo punto, como un?)
{:.center}
![linear classifier](/images/lin_classifier_2d.png 'linear classifier')

Aparece en el lado negativo de nuestro límite de decisión, y por lo tanto predecimos que no lloverá. Más concretamente, nuestra decisión de clasificación se puede expresar como:

$$
\begin{eqnarray}
\mbox{classification} & = & \left\{ \begin{array}{ll}
1 & \mbox{if } w_1*x_1 + w_2*x_2 + b \gt 0 \\
0 & \mbox{if } w_1*x_1 + w_2*x_2 + b \leq 0
\end{array} \right.
\tag{1}\end{eqnarray}
$$

Este es un clasificador lineal bidimensional.

Ahora hagamos lo mismo en 3 dimensiones. Al agregar una columna obtenemos esto:

{% include video.html mp4='/images/video.mp4' webm='/images/video.webm' width='400' %}

Un plano plano en 3d es análogo a una línea en 2d, y por eso se llama \"lineal.\" Esto es cierto en general para cualquier hiperplano n-dimensional. Los clasificadores lineales son limitados porque, en realidad, la mayoría de los problemas que nos interesan no tienen un comportamiento tan plano; diferentes variables interactúan de varias maneras.

{:.section}
Dimensión X

En la práctica, tenemos muchas dimensiones, pero funciona igual.

{:.section}
Limitaciones del clasificador lineal

A veces nuestros datos no son linealmente separables. Supongamos que recibimos un conjunto de entrenamiento que se ve así: puntos en el medio.
Claramente, ninguna línea va a
[3] - 2d no separable linealmente

# sin etiquetas

Más tarde, sin supervisión

# Conectando los puntos


Puede parecer difícil de creer, pero esta configuración simple forma el núcleo de

el mismo clasificador lineal que simplemente separa dos objetos entre sí es lo que sustenta las babosas de los cachorros, bots de shakespeare, incrustaciones de palabras (word2vec) y otros. Puede parecer difícil de creer al principio, pero se basa en un punto sutil. Cuando entrenamos un algoritmo para discernir la relación entre un conjunto de observaciones y un comportamiento correspondiente, estamos haciendo más que dejarlo hacer nuevas predicciones. Estamos también formando una _representación_ de nuestro tema de interés, un esquema computacional.

** ttomados en conjunto, los pesos y los sesgos a menudo también se denominan _parámetros_ porque el comportamiento de nuestra máquina depende de cómo los configuremos.

# Aprendizaje supervisado

Esto es aprendizaje supervisado. utilizado para bla

## Regresión: ejemplo simple

Para hacer las cosas más concretas, echemos un vistazo al ejemplo más simple de aprendizaje automático: la regresión lineal. Si alguna vez has tomado un curso de estadística en la escuela secundaria, ¡probablemente los hayas resuelto a mano!

La regresión lineal es una técnica utilizada para descubrir la relación funcional subyacente entre

Regresión lineal en 2d
Regresión lineal en 2d (p5?)

Variaciones
- Regresión/clasificación logística

En la práctica, la regresión lineal casi nunca se usa porque la mayoría de las funciones que nos interesan no son tan simples y se requieren métodos más complejos.

- bishop
- Sobreajuste (overfitting)

# Aprendizaje no supervisado

Encuentra la estructura subyacente

# Aprendizaje por refuerzo (Reinforcement learning)

demostración física (balancing a stick) (top banner?). este libro probablemente hará referencia principalmente

-----

from: http://ufldl.stanford.edu/tutorial/supervised/LinearRegression/

For example, we might want to make predictions about the price of a house so that represents the price of the house in dollars and the elements of represent “features” that describe the house (such as its size and the number of bedrooms). Suppose that we are given many examples of houses where the features for the i’th house are denoted and the price is . For short, we will denote the

Our goal is to find a function
succeed in finding a function
prices, we hope that the function
given the features for a new house where the price is not known.
so that we have for each training example. If we like this, and we have seen enough examples of houses and their will also be a good predictor of the house price even when we are
given the features for a new house where the price is not known.

We initialize a sigmoid neural network with 3 input neurons and 1 output neuron, and 1 hidden layer with 2 neurons. Every connection has a random initial weight, and neurons in the hidden and output layers have a random bias.
