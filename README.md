# Artificial Intelligence: Reinforcment Learning

Academic project (Fall 2019) to explore an AI concept.

I chose to research and implement an reinforcement learning agent to play the Atari game, Freeway. Using [OpenAI Gym](https://gym.openai.com/)

Research included trying self-built implementations of model-based reinforcement learning and then Approximate Q learning. In the end, it was determined a more complex solution was needed. This resulted in using Deep Q Learning and Double Deep Q Learning. See project analysis in Analysis.pdf. 

The Double Deep Q implementation was built using existing [source code](https://github.com/abhinavsagar?utf8=%E2%9C%93&tab=repositories&q=&type=&language=) and [tutorial](https://towardsdatascience.com/deep-reinforcement-learning-tutorial-with-open-ai-gym-c0de4471f368). See the project refactored, implemented version that trains and plays Atari Freeway in [DDQN Freeway Refactoring](https://github.com/DFinelli/AtariFreeway-DeepLearning/tree/master/DDQN%20Freeway%20Refactoring).

Freeway is an atari agent where a chicken must cross the road while dodging traffic. 

![Freeway](https://github.com/DFinelli/AtariFreeway-DeepLearning/blob/master/freeway.gif)
#
[Image Source](https://www.retrogames.cz/play_123-Atari2600.php)
