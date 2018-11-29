Technical documentation
=======================

This page describes the technical implementation of mcfly and the choices that have been made.

Hyperparameter search
---------------------
Mcfly performs a random search over the hyper parameter space (see the section in the user manual about which hyperparameters are tuned). 
We chose to implement random search, because it's simple and fairly effective. We considered some alternatives:

* Bayesian optimization with Gaussian processes, such as `spearmint <https://github.com/HIPS/Spearmint>`_: 
  is not usable for a mix of discrete (e.g. number of layers) and continuous hyperparameters
* Tree of Parzen Estimator (TPE, implemented in `hyperopt <http://hyperopt.github.io/hyperopt/>`_) is a Bayesian optimization method that can be used for 
  discrete and conditional hyperparameters. 
  Unfortunately, hyperopt is not actively maintained and the latest release is not python 3 compatible. 
  (NB: the package `hyperas <https://github.com/maxpumperla/hyperas>`_ provides a wrapper around hyperopt, specifically for Keras)
* `SMAC <http://www.cs.ubc.ca/labs/beta/Projects/SMAC/>`_ is a hyperparameter optimization method that uses Random Forests to sample the new distribution. 
  We don't use SMAC because the python package depends on a Java program (for which we can't find the source code).

If you are interested in the different optimization methods, we recommend the following readings:

* Bergstra, James S., et al. "Algorithms for hyper-parameter optimization." Advances in Neural Information Processing Systems. 2011. (`link <https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf>`_)
* Hutter, Frank, Holger H. Hoos, and Kevin Leyton-Brown. "Sequential model-based optimization for general algorithm configuration." 
  International Conference on Learning and Intelligent Optimization. Springer Berlin Heidelberg, 2011. (`link <http://www.cs.ubc.ca/labs/beta/Projects/SMAC/papers/11-LION5-SMAC.pdf>`_)
* Eggensperger, Katharina, et al. "Towards an empirical foundation for assessing bayesian optimization of hyperparameters." 
  NIPS workshop on Bayesian Optimization in Theory and Practice. 2013. (`link <http://aad.informatik.uni-freiburg.de/papers/13-BayesOpt_EmpiricalFoundation.pdf>`_)
* `Blogpost <http://www.argmin.net/2016/06/20/hypertuning/>`_ by Ben Recht
* `Blogpost <http://blog.turi.com/how-to-evaluate-machine-learning-models-part-4-hyperparameter-tuning>`_ by Alice Zheng


Architectures
-------------
There are two types of architectures that are available in mcfly: CNN and DeepConvLSTM. 
The first layer in both architectures is a Batchnorm layer (not shown below), so that the user doesn't have to normalize the data during data preparation.

CNN
^^^
The model type CNN is a 'regular' Convolutional Neural Network, with N convolutional layers with Relu activation and one hidden dense layer. So the architecture looks like:

``Batchnorm - [Conv - Batchnorm - Relu]*N - Dense - Relu - Dense - Batchnorm - Softmax``

The number of Conv layers, as well as the number of filters in each Conv layer and the number of neurons in the hidden Dense layer are hyperparameters of this model. We decided not to add Pool layers because reducing the spatial size of the sequence is usually not necessary if you have enough convolutional layers.

DeepConvLSTM
^^^^^^^^^^^^
The architecture of the model type DeepConvLSTM is based on the paper: Ordóñez et al. (2016). The architecture looks like this:

``Batchnorm - [Conv - Relu]*N - [LSTM]*M - Dropout - TimeDistributedDense - Softmax - TakeLast``

The Softmax layer outputs a sequence of predictions, so we need a final TakeLast layer (not part of Keras) to pick the last element from the sequence as a final prediction. In contrast to the CNN model, the convolutional layers in the DeepConvLSTM model are applied per channel, and only connected in the first LSTM layer. The hyperparameters are the number of Conv layers, the number of LSTM layers, the number of filters for each Conv layer and the hidden layer dimension for each LSTM layer. Note that in the paper of Ordóñez et al, the specific architecture has 4 Conv layers and 2 LSTM layers.

Other choices
-------------
We have made the following choices for all models:

* We use LeCun Uniform weight initialization (`LeCun 1998 <http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf>`_)
* We use L2 regularization on all convolutional and dense layers
* We use categorical cross-entropy loss
* We output accuracy and take this as a measure to choose the best performing model

Comparison with non-deep models
---------------------------------
To check the value of the data, a 1-Nearest Neighbors model is applied as a benchmark for the deep learning model. 
We chose 1-NN because it's a very simple, hyperparameter-free model that often works quite well on time series data. 
For large train sets, 1-NN can be quite slow: the test-time performance scales linear with the size of the training set. 
However, we perform the check only on a small subset of the training data. 
The related Dynamic Time Warping (DTW) algorithm has a better track record for classifying time series, 
but we decided not to use it because it's too slow (it scales quadratically with the length of the time series).