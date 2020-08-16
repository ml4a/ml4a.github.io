---
layout: chapter
title: "Game playing"
---

quote
 - from the glass bead game

http://www.starcenter.com/glassbead.pdf


http://www.nervanasys.com/demystifying-deep-reinforcement-learning/

http://www.somatic.io/blog/on-alphago-intuition-and-the-master-objective-function?utm_content=buffera3f62&utm_medium=social&utm_source=twitter.com&utm_campaign=buffer

https://medium.com/a-year-of-artificial-intelligence/lenny-2-autoencoders-and-word-embeddings-oh-my-576403b0113a#.q9cfvzmpi


yan tweet: RL is the cherry on the cake

A Game, for example, might start from a given astronomical configuration, or from the actual theme of a Bach fugue, or from a sentence out of Leibniz or the Upanishads, and from this theme, depending on the intentions and talents of the player, it could either further explore and elaborate the initial motif or else enrich its expressiveness by allusions to kindred concepts... It represented an elite, symbolic form of seeking for perfection, a sublime alchemy, an approach to that Mind which beyond all images and multiplicities is one within itself -- in other words, to God.
Herman Hesse, The Glass Bead Game

I will do my best to play a beautiful and interesting game.
Lee Se-dol


Tic tac toe -> DeepBlue -> AlphaGo

==========

http://www.nervanasys.com/demystifying-deep-reinforcement-learning/

http://googleresearch.blogspot.com/2016/01/alphago-mastering-ancient-game-of-go.html

Go - alpha mind vs lee sedol

reinforcement learning
 - games that learn by pixels

even harder
 - doom

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


In March 2016, an AI trained to play [Go](https://en.wikipedia.org/wiki/Go_(game)) defeated Lee Se-dol, the 9-dan champion regarded by many as the best player in the history of the ancient board game. For many years, Go was thought intractable for AIs to compete with the top human players. Chess computers had caught up to grandmasters like Gary Kasparov some 20 years before, as when IBM's DeepBlue beat the world's top chess player at the time. But it was clear that DeepBlue's approach could not be scaled to Go. For one thing, the number of legal board positions in Go is vastly greater than in chess. Chess is estimated to have $$10^{120}$$ possible board positions, already far more than the $$10^80$$ atoms in the universe. Go has $$10^{761}$$ possible boards, or $$10^641$$ times _as many as chess_! This inconceivably large number of positions means that brute force search of all possible moves will not work.

As we shall see, AlphaGo, built by Google's DeepMind group, uses an ensemble of techniques from machine learning to create an  ???, including a duo of convolutional neural networks, which we will get to later. But it uses it in tandem with some novel techniques we haven't covered yet. In particular, it uses a form of _reinforcement learning_ to self-improve from experience, a technique commonly used to build AIs which learn how to play and win video games.

Perhaps the most striking aspect of the AlphaGo system is its near-total reliance on very general algorithms which are not initially given any expert knowledge about Go. This is in contrast to DeepBlue which relied on a hand-crafted evaluation function to measure the utility of board positions. Thus, AlphaGo-like programs may be easier to adapt to other games, or more general scenarios altogether.

One of the most clever ways AlphaGo achieved this is by repeatedly playing against itself and using reinforcement learning to improve its ability. This chapter will first introduce reinforcement learning in the context of video game-playing AIs and return to the case of AlphaGo later.

## Gaming

**[Figure:: Atari games]**

Initially, the choice of video games as a context for an AI task seems puzzling, as it may seem of little interest to people who aren't gamers. But it turns out it's not video games _per se_ which are of interest to AI researchers, it's a much more general problem we are studying, and video games happen to be an effective way to represent it.

The general problem is of some computational agent in some initially unknown environment, trying to learn -- on the fly -- how to interact with this environment to maximize some reward. Perhaps the agent is trying to navigate this space, locate something valuable in it, or control some unfolding process in some value-maximizing way. This scenario abstractly represents countless real-world applications, and indeed, reflects a type of learning that humans demonstrate rather well. From the time of birth, we often find ourselves in unknown environments, learning from observation, how it responds to our manipulations. This is very different from the kinds of problems this book has covered so far, where all the information was provided up front. 

Atari materials
 - /Users/gene/bin/misc/stock/Google DeepMind's Deep Q-learning playing Atari Breakout-V1eYniJ0Rnk.mp4

## Reinforcement learning

If intelligence was a cake, unsupervised learning would be the cake, supervised learning would be the icing on the cake, and reinforcement learning would be the cherry on the cake. We know how to make the icing and the cherry, but we don't know how to make the cake.

https://www.facebook.com/yann.lecun/posts/10153426023477143

As a child, you learn not to touch things that are hot from just one teary-eyed encounter. To a computer, it's not clear what "hot" even means, which makes writing a program that learns it from one experience challenging. Even more impressively, infants acquire the meaning of words upon just one exposure to hearing them. We take this astonishing human faculty for granted. In contrast, neural networks require thousands of examples of audio just to be able to determine what was said, let alone what it meant. 

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



alphago slides





96-97 chess
 - how old?
 - moves felt like it had purpose
 - but chess wasn't actually solved yet, kasparov played wrong


https://www.quantamagazine.org/20160329-why-alphago-is-really-such-a-big-deal/
https://www.tastehit.com/blog/google-deepmind-alphago-how-it-works/

 policy network -> self-playing -> value network
 pn + vn => MCTS -> pick move
 pn + vn = convnets!
 pn plays against itself


1) policy network
  - trained on millions of games
    - [ x0x0x00x ] (go Board) => 1-hot vector of new element
  - predict next move 57% of the time
  - plays Go at amateur level
  - con: no way of evaluating value of position

2) value network
   - pipe in board position, get estimate of value
   - trained on many games again
   - improved through self-play
   - wild! self-improvement

3) MCTS
   - need to narrow search space
   - policy network gives candidate moves (high probability next)
   - seek to some depth and eval with value network
   - pick move


Cell based
 - cell-based board games, basic problem setup
   - chess, checkers, go, 
   - it's a tree search problem

 - one really nice thing about the AlphaGo program is it uses an ensemble of techniques including convnets which are general enough to give insights about problems that are similarly structured

 - tree search
 - monte carlo tree search
 - at first naive: just check every possible move, and check if you win
   - 10^80 games --> too many to evaluate (more than atoms in universe)
 - so with chess, you search only some small number of steps, and use some heuristics
   - with chess, super complicated
     - easy would be to just count pieces, or more complex, count cells being threatened by your pieces
     - with IBM, it was like 8000 rules, really expensive to implement
     - this means it doesn't generalize well, you have to design new rules for different games

 - cool thing about AlphaGo is it uses convolutional neural networks to replace the creation of hand-crafted rules
   - this is great because it's much more accurate than hand-crafted rules (so we need to make less evaluations)
   - even better for AGI because it generalizes to other kinds of problems

 - more details


 - it gets harder
   - go is "information complete", i.e. you have all the info you need looking at the current gameboard and no concept of history is needed
   - in other games, this is not true. in Doom/quake, we don't see everything at the same time, location of info changes as we change perspective. 


artificial general intelligence

Unfortunately, Nature doesn't make science available to the public, but you can download the paper anyway at various links suggested in this Reddit thread.

Reinforcement learning
 - Mnih vs. Atari [http://home.uchicago.edu/~arij/journalclub/papers/2015_Mnih_et_al.pdf] [https://www.youtube.com/watch?v=iqXKQf2BOSE]

 - Mario NEAT (genetic algos) [https://www.youtube.com/watch?v=qv6UVOQ0F44]
 - Mario fun https://www.youtube.com/watch?v=xOCurBYI_gY

Balancing pole
 - https://www.youtube.com/watch?v=Lt-KLtkDlh8

FlappyBird
 - https://www.youtube.com/watch?v=xM62SpKAZHU

pong

AlphaGO

Quake, Doom, etc

DeepBlue
 - Kasparov v world

Artificial general intelligence
 - https://en.wikipedia.org/wiki/AI_takeover

How AlphaGo works
 - tic-tac-toe + tree search
 - chess + Monte Carlo Tree Search
 - AlphaGo
   - convnets + RL + MCTS

Reinforcement learning
 - https://openai.com/blog/openai-gym-beta/


https://www.quantamagazine.org/20160329-why-alphago-is-really-such-a-big-deal/


Poker playing
 - https://www.reddit.com/r/IAmA/comments/5qi3i9/we_are_professional_poker_players_currently/

http://www.nervanasys.com/demystifying-deep-reinforcement-learning/

challenges 
 - delayed reward + credit assignment
 - discount factor: exploration vs exploitation


RL agent swinging a pole
 - swingbot-reinforcement-learning

http://www.theverge.com/2016/3/10/11192774/demis-hassabis-interview-alphago-google-deepmind-ai

john schulman https://www.youtube.com/watch?v=oPGVsoBonLM

agent cooperation https://deepmind.com/blog/understanding-agent-cooperation/

nakamura vs rybka https://www.chess.com/article/view/computers-in-chess-good-or-evil-part-two

https://pathak22.github.io/noreward-rl/resources/icml17.pdf

https://pathak22.github.io/noreward-rl/

https://medium.com/applied-data-science/how-to-build-your-own-alphazero-ai-using-python-and-keras-7f664945c188