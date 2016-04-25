---
layout: post
title: "Game playing"
date: 2016-05-02
---

quote
 - from the glass bead game

Go - alpha mind vs lee sedol

reinforcement learning
 - games that learn by pixels

UETorch


alphago - https://www.youtube.com/watch?v=7f47A7Pd11k&feature=youtu.be

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


In March 2016, an AI designed to play the ancient board game of Go defeated Lee Se-dol, the 9-dan widely regarded as the best player in the world. For years, Go -- an ancient board game popular all over the world, particularly in East Asia -- was thought intractable for AIs to compete with human masters. Although chess had been solved some 20 years before, when IBM's DeepBlue beat Gary Kasparov, the #1 chess player in the world at the time, it was clear that its approach could not be adapted to Go. For one thing, the number of legal board positions in Go is vastly greater than in chess. Chess is estimated to have $$10^{80}$$ possible board positions, already more than the number of atoms in the universe. Go on the other hand, has $$10^{40}$$ times _as many as chess_! This inconceivably large number of positions means that exhaustive search of possible moves would not suffice.

As we shall see, this new Go champion, AlphaGo built by Google's DeepMind, uses an ensemble of techniques in machine learning to create an effective game-playing algorithm, including a duo of convolutional neural networks, which we will get to later. But it uses it in tandem with some novel techniques we haven't covered yet. In particular, it uses a form of _reinforcement learning_ to self-improve from experience, a technique commonly used to build AIs which learn how to play and win video games.

Before returning to AlphaGo, this chapter will introduce reinforcement learning in the context of game-playing AIs.

## Gaming

**[Figure:: Atari games]**

Initially, the choice of video games as a context for an AI task seems puzzling, as it may seem of little interest to people who aren't gamers. But it turns out it's not video games _per se_ which are of interest to AI researchers, it's a much more general scenario we are studying, and video games happen to be an effective way to represent it.

The general problem is of some computational agent in some initially unknown environment, trying to learn -- on the fly -- how to negotiate this environment and maximize some reward. Perhaps the agent is trying to navigate this space, locate something valuable in it, or control (?) some unfolding process in the most efficient or value-maximizing way possible. At any rate, this scenario abstractly represents countless many real-world applications, and indeed, reflects a more general type of learning that humans know well. After all, from the time of birth, we often find ourselves in unknown environments, learning from observation how it responds to our manipulations. This is very different from the kinds of problems this book has covered so far, in which case all the information would be provided up front. 

## Reinforcement learning

Atari games, space invaders, mario


# AlphaGo

Getting back to AlphaGo, we have now covered two of the three major pre-requisites necessary to understand how AlphaGo works: reinforcement learning (above) and [convolutional neural networks](_convnets_). The last component is [Monte Carlo tree search](_MCTS_), which we will build up to in the context of two other board games, tic-tac-toe and chess.



We now 

MCTS
 - tic tac toe
 - chess
 - go


