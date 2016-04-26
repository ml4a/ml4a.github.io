---
layout: post
title: "Game playing"
date: 2016-05-02
---

quote
 - from the glass bead game

http://www.starcenter.com/glassbead.pdf


A Game, for example, might start from a given astronomical configuration, or from the actual theme of a Bach fugue, or from a sentence out of Leibniz or the Upanishads, and from this theme, depending on the intentions and talents of the player, it could either further explore and elaborate the initial motif or else enrich its expressiveness by allusions to kindred concepts... It represented an elite, symbolic form of seeking for perfection, a sublime alchemy, an approach to that Mind which beyond all images and multiplicities is one within itself -- in other words, to God.
Herman Hesse, The Glass Bead Game

I will do my best to play a beautiful and interesting game.
Lee Se-dol

==========

http://googleresearch.blogspot.com/2016/01/alphago-mastering-ancient-game-of-go.html

Go - alpha mind vs lee sedol

reinforcement learning
 - games that learn by pixels

UETorch


alphago - https://www.youtube.com/watch?v=7f47A7Pd11k&feature=youtu.be
https://www.quantamagazine.org/20160329-why-alphago-is-really-such-a-big-deal/

Lee
I, Lee Se-dol, lost, but mankind did not http://www.koreatimes.co.kr/www/news/tech/2016/03/133_200267.html

"I will do my best to play a beautiful and interesting game,"
http://www.cnbc.com/2016/03/08/google-deepminds-alphago-takes-on-go-champion-lee-sedol-in-ai-milestone-in-seoul.html

"AlphaGo made me realize that I must study Go more,”
http://www.nytimes.com/2016/03/16/world/asia/korea-alphago-vs-lee-sedol-go.html

“It made me question human creativity. When I saw AlphaGo’s moves, I wondered whether the Go moves I have known were the right ones,” the human competitor, Lee Se-dol, 33, said during a postmatch news conference. “Its style was different, and it was such an unusual experience that it took time for me to adjust.”

“AlphaGo made me realize that I must study Go more,” said Mr. Lee, one of the world’s most accomplished players.
http://www.nytimes.com/2016/03/16/world/asia/korea-alphago-vs-lee-sedol-go.html


AlphaGo

https://www.quantamagazine.org/20160329-why-alphago-is-really-such-a-big-deal/


In March 2016, an AI endowed with the ability to play [Go](https://en.wikipedia.org/wiki/Go_(game)) defeated Lee Se-dol, the 9-dan champion regarded by many as the best player in the history of the ancient board game. For many years, Go was thought intractable for AIs to compete with the top human players. Chess computers had caught up to grandmasters like Gary Kasparov some 20 years before, as when IBM's DeepBlue beat the world's top chess player at the time. But it was clear that DeepBlue's approach could not be adapted to Go. For one thing, the number of legal board positions in Go is vastly greater than in chess. Chess is estimated to have $$10^{120}$$ possible board positions, already far more than the $$10^80$$ atoms in the universe. Go has $$10^{761}$$ possible boards, or $$10^641$$ times _as many as chess_! This inconceivably large number of positions means that brute force search of all possible moves will not work.

As we shall see, AlphaGo, built by Google's DeepMind group, uses an ensemble of techniques in machine learning to create an  ____, including a duo of convolutional neural networks, which we will get to later. But it uses it in tandem with some novel techniques we haven't covered yet. In particular, it uses a form of _reinforcement learning_ to self-improve from experience, a technique commonly used to build AIs which learn how to play and win video games.

Perhaps the most striking aspect of the AlphaGo system is its near-total reliance on very general algorithms which are not initially given any expert knowledge about Go. This is in contrast to DeepBlue which relied on a hand-crafted evaluation function to measure the utility of board positions. Thus, AlphaGo-like programs may be easier to adapt to other kinds of games, or more general scenarios altogether.

One of the most clever ways AlphaGo achieved this is by repeatedly playing against itself and using reinforcement learning to improve its ability. This chapter will first introduce reinforcement learning in the context of video game-playing AIs and return to the case of AlphaGo later.

## Gaming

**[Figure:: Atari games]**

Initially, the choice of video games as a context for an AI task seems puzzling, as it may seem of little interest to people who aren't gamers. But it turns out it's not video games _per se_ which are of interest to AI researchers, it's a much more general scenario we are studying, and video games happen to be an effective way to represent it.

The general problem is of some computational agent in some initially unknown environment, trying to learn -- on the fly -- how to interact with this environment to maximize some reward. Perhaps the agent is trying to navigate this space, locate something valuable in it, or control some unfolding process in some value-maximizing way. This scenario abstractly represents countless real-world applications, and indeed, reflects a more general type of learning that humans demonstrate rather well. After all, from the time of birth, we often find ourselves in unknown environments, learning from observation, how it responds to our manipulations. This is very different from the kinds of problems this book has covered so far, where all the information was provided up front. 

## Reinforcement learning

If intelligence was a cake, unsupervised learning would be the cake, supervised learning would be the icing on the cake, and reinforcement learning would be the cherry on the cake. We know how to make the icing and the cherry, but we don't know how to make the cake.

https://www.facebook.com/yann.lecun/posts/10153426023477143

As a child, you learn not to touch things that are hot from just one teary-eyed encounter. To a computer, it's not clear what "hot" even means, which makes writing a program that learns it from one experience challenging. Even more impressively, infants acquire the meaning of words from just one exposure to hearing it. We take this astonishing human faculty for granted. In contrast, neural networks require thousands of examples of audio just to be able to determine what was said, let alone what it meant. 

This makes programming agents to figure out how to play and beat video games very difficult. We are usually given nothing more than the pixels of the screen, and asked to provide a sequence of actions which eventually win the game. For games with the complexity of Super Mario or __, this is especially daunting. But we can start with a much more simple game: balancing a rod. While this game is unlikely to outsell Halo anytime soon, it provides a simple version of a game with all of our criteria. 

[ balancing ]

Atari games, space invaders, mario


# From Tic-Tac-Toe to Chess to Go

There remains one more component we need to understand before we put all the pieces of AlphaGo together, and that is Monte Carlo Tree Search (MCTS). MCTS is an algorithm which chooses the candidate moves to evaluate, something which will be necessary for both Chess and Go, since the number of possible moves is too large to try out completely. We'll build up to MCTS by considering tic-tac-toe first.

We can imagine organizing all of the possible games of tic-tac-toe, as a tree with a certain depth.

**[ Tree ]**

In tic-tac-toe, the number of nodes in this tree is 765. An algorithm could easily parse this tree, and count the most likely path towards a win at each step. But when we consider the case of chess, which can also be represented as a tree of possible game sequences, we can no longer do this because the space of possible moves is too large. 

The solution to this problem is to 

# Putting it all together: AlphaGo

We have now covered two of the three major pre-requisites necessary to understand how AlphaGo works: reinforcement learning (above) and [convolutional neural networks](_convnets_). The last component is [Monte Carlo tree search](_MCTS_), which we will build up to in the context of two other board games, tic-tac-toe and chess.


https://www.tastehit.com/blog/google-deepmind-alphago-how-it-works/

http://googleresearch.blogspot.com/2016/01/alphago-mastering-ancient-game-of-go.html
http://www.slideshare.net/ShaneSeungwhanMoon/how-alphago-works
https://www.dcine.com/2016/01/28/alphago/
https://www.reddit.com/r/artificial/comments/4d183c/michael_nielsen_is_alphago_really_such_a_big_deal/
https://www.quantamagazine.org/20160329-why-alphago-is-really-such-a-big-deal/

https://en.wikipedia.org/wiki/Reinforcement_learning


https://www.youtube.com/watch?v=ifma8G7LegE

visual doom
http://vizdoom.cs.put.edu.pl/competition-cig-2016


reddit
https://www.reddit.com/r/MachineLearning/comments/42ytdx/pdf_mastering_the_game_of_go_with_deep_neural/

We now 

MCTS
 - tic tac toe
 - chess
 - go


