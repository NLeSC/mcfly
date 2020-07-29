User Manual
===========

On this page, we describe what you should know when you use mcfly. This manual should be understandable without too much knowledge of deep learning,
although it expects familiarity with the concepts of dense hidden layers, convolutional layers and recurrent layers.
However, if mcfly doesn't give you a satisfactory model, a deeper knowledge of deep learning really helps in debugging the models.

We provide a quick description for the layers used in mcfly.

* **dense layer** also know as fully connected layer, is a layer of nodes that all have connections to all outputs of the previous layer.
* **convolutional layer** convolves the output of the previous layer with one or more sets of weights and outputs one or more feature maps.
* **LSTM layer** is a recurrent layer with some special features to help store information over multiple time steps in time series.

Some recommended reading to make you familiar with deep learning:
http://scarlet.stanford.edu/teach/index.php/An_Introduction_to_Convolutional_Neural_Networks

Or follow a complete course on deep learning:
http://cs231n.stanford.edu/


The function findBestArchitecture
---------------------------------
The function :func:`~mcfly.find_architecture.find_best_architecture` generates a variety of architectures and hyperparameters,
and returns the best performing model on a subset of the data.
The following four types of architectures are possible (for a more detailed description of the models, see the :doc:`technical_doc`):

**CNN**: A stack ofonvolutional layers, followed by a final dense layer

**DeepConvLSTM**: Convolutional layers, followed by LSTM layers and a final dense layer

**ResNet**: Convolutional layers with skip connections

**InceptionTime**: Convolutional layers ('inception module') with different kernel sizes in parallel, concatenated and then followed by pooling and a dense layer.

The hyperparameters to be optimized are the following:

* learning rate
* regularization rate
* if modeltype=CNN:
   * number of Conv layers
   * for each Conv layer: number of filters
   * number of hidden nodes for the hidden Dense layer

* if modeltype=DeepConvLSTM:
   * number of Conv layers
   * for each Conv layer: number of filters
   * number of LSTM layers
   * for each LSTM layer: number of hidden nodes

* if modeltype=ResNet:
   * network depth, i.e. number of residual modules
   * minimum number of filters
   * maximum kernel size

* if modeltype=InceptionTime:
   * number of filters for all convolutional layers
   * depth of network, i.e. number of Inception modules to stack.
   * maximum kernel size


We designed mcfly to have sensible default values and ranges for each setting.
However, you have the possibility to influence the behavior of the function with the arguments that you give to it to try other values.
These are the options (see also the documentation of :func:`~mcfly.modelgen.generate_models`):

* **number_of_models**: the number of models that should be generated and tested
* **nr_epochs**: The models are tested after only a small number of epochs, to limit the time. Setting this number higher will give a better estimate of the performance of the model, but it will take longer
* **model_type** Specifies which type of model ('CNN' or 'DeepConvLSTM') to generate. With default value None it will generate both CNN and DeepConvLSTM models.
* Ranges for all of the hyperparameters: The hyperparameters (as described above) are sampled from a uniform or log-uniform distribution. The boundaries of these distributions have default values (see the arguments :func:`~mcfly.modelgen.generate_models`), but can be set custom.


Visualize the training process
-------------------------------
To gain more insight in the training process of the models and the influence of the hyperparameters, you can explore the visualization.

1. Save the model results, by defining `outputpath` in `find_best_architecture`.

2. Start an python webserver (see :doc:`installation`) and navigate to the visualization page in your browser.

3. Open the json file generated in step 1.

In this visualization, the accuracy on the train and validation sets are plotted for all models. You can filter the graphs by selecting specific models, or filter on hyperparameter values.

FAQ
---

None of the models that are tested in findBestArchitecture perform satisfactory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Note that :func:`~mcfly.find_architecture.find_best_architecture` doesn't give you a fully trained model yet: it still needs to be trained on the complete dataset with sufficient iterations.
However, if none of the models in :func:`~mcfly.find_architecture.find_best_architecture` have a better accuracy than a random model, it might be worth trying one of the following things:

* Train more models: the number of models tested needs to be sufficient to cover a large enough part of the hyperparameter space
* More epochs: it could be that the model needs more epochs to learn (for example when the learning rate is small). Sometimes this is visible from the learning curve plot
* Larger subset size: it could be that the subset of the train data is too small to contain enough information for learning
* Extend hyperparameter range
