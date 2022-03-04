# Actor-Critic

Here will my Advantage Actor-Critic (A2C) implementation live. `models.py` contains different types of actor critic network architectures, e.g. a two separate fully connected networks, a fully-connected shared backbone and a convolutional shared backbone. `a2c.py` contains the agent class and the methods for training the agent. Currently `a2c.py` is tested only on the `CartPole` environment.

## TODOs

* training script for sychronous training and runs with different hyperparameter settings
* add GAE
* test on atari environments


## References 

* [Asychronous Methods for Deep Reinforcement Learning](https://arxiv.org/pdf/1602.01783.pdf)
* [Actor-Critic Algorithms](http://rail.eecs.berkeley.edu/deeprlcourse-fa17/f17docs/lecture_5_actor_critic_pdf)
