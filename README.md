# Artificial Intelligence: Reinforcment Learning

Academic project (Fall 2019) completed as a team of 2. The assigned task was to research an AI concept and try impementations.

Our project goal was to research and implement a reinforcement learning agent to train and play the Atari game, Freeway using [OpenAI Gym](https://gym.openai.com/).

First, self-built Model-Based Reinforcement Learning implementations proved unsuccessful. Then, Approximate Q Learning implementations failed. In the end, it was determined a more complex solution was needed: Deep Q Learning and Double Deep Q Learning. See full project analysis in ![Analysis](https://github.com/DFinelli/AtariFreeway-DeepLearning/blob/master/Analysis.pdf).

The Double Deep Q implementation was built using existing [source code](https://github.com/abhinavsagar?utf8=%E2%9C%93&tab=repositories&q=&type=&language=) following a [tutorial](https://towardsdatascience.com/deep-reinforcement-learning-tutorial-with-open-ai-gym-c0de4471f368). The final implimenation refactored code, originally built for Space Invaders, to train and play Freeway. See implementation in [DDQN Freeway Refactoring](https://github.com/DFinelli/AtariFreeway-DeepLearning/tree/master/DDQN%20Freeway%20Refactoring).

Freeway is an Atari game where a chicken must cross the road while dodging traffic in the allotted time.

![Freeway](https://github.com/DFinelli/AtariFreeway-DeepLearning/blob/master/freeway.gif)
#
[Image Source](https://www.retrogames.cz/play_123-Atari2600.php)
