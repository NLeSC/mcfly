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


Data preprocessing
-------------------

The input for the mcfly functions is data that is already preprocessed to be handled for deep learning module Keras.
In this section we describe what data format is expected and what to think about when preprocessing.

Eligible data sets
^^^^^^^^^^^^^^^^^^
Mcfly is a tool for *classification* of single or *multichannel timeseries data*. One (real valued) multi-channel time series is associated with one class label.
All sequences in the data set should be of equal length.

Data format
^^^^^^^^^^^
The data should be split in train, validation and test set. For each of the splits, the input X and the output y are both numpy arrays.

The input data X should be of shape (num_samples, num_timesteps, num_channels). The output data y is of shape (num_samples, num_classes), as a binary array for each sample.

We recommend storing the numpy arrays as binary files with the numpy function ``np.save``.

Data preprocessing
^^^^^^^^^^^^^^^^^^
Here are some tips for preprocessing the data:

* For longer, multi-label sequences, we recommend creating subsequences with a sliding window. The length and step of the window can be based on domain knowledge.
* One label should be associated with a complete time series. In case of multiple labels, often the last label is taken as the label for the complete sequence.
  Another possibility is to take the majority label.
* In splitting the data into training, validation and test sets, it might be necessary to make sure that sample subject (such as test persons) for which multiple sequences are available, are not present in both train and validation/test set. The same holds for subsequences that originates from the same original sequence (in case of sliding windows).
* The Keras function ``keras.utils.np_utils.to_categorical`` can be used to transform an array of class labels to binary class labels.
* Data doesn't need to be normalized. Every model mcfly produces starts by normalizing data through a Batch Normalization layer.
  This means that training data is used to learn mean and standard deviation of each channel and timestep.

Finding the best architecture
---------------------------------
The function :func:`~mcfly.find_architecture.find_best_architecture` generates a variety of architectures and hyperparameters,
and returns the best performing model on a subset of the data.
The following four types of architectures are possible (for more information, see the :doc:`technical_doc`):

:class:`~mcfly.models.CNN`: A stack ofonvolutional layers, followed by a final dense layer

:class:`~mcfly.models.ConvLSTM`: Convolutional layers, followed by LSTM layers and a final dense layer

:class:`~mcfly.models.ResNet`: Convolutional layers with skip connections

:class:`~mcfly.models.InceptionTime`: Convolutional layers ('inception module') with different kernel sizes in parallel, concatenated and then followed by pooling and a dense layer.

The hyperparameters to be optimized are the following:

* learning rate
* regularization rate
* model_type: *CNN* or *DeepConvLSTM*
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
See the the documentation of :func:`~mcfly.modelgen.generate_models` for all options, among others:
* **number_of_models**: the number of models that should be generated and tested
* **nr_epochs**: The models are tested after only a small number of epochs, to limit the time. Setting this number higher will give a better estimate of the performance of the model, but it will take longer
* **model_types**: List of all model architecture types to choose from
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
