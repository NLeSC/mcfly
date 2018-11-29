Data preprocessing
==================

The input for the mcfly functions is data that is already preprocessed to be handled for deep learning module Keras. 
On this page, we describe what data format is expected and what to think about when preprocessing.

Eligible data sets
-------------------
Mcfly is a tool for *classification* of single or *multichannel timeseries data*. One (real valued) multi-channel time series is associated with one class label. 
All sequences in the data set should be of equal length.

Data format
------------
The data should be split in train, validation and test set. For each of the splits, the input X and the output y are both numpy arrays.

The input data X should be of shape (num_samples, num_timesteps, num_channels). The output data y is of shape (num_samples, num_classes), as a binary array for each sample.

We recommend storing the numpy arrays as binary files with the numpy function ``np.save``.

Data preprocessing 
------------------
Here are some tips for preprocessing the data:

* For longer, multi-label sequences, we recommend creating subsequences with a sliding window. The length and step of the window can be based on domain knowledge.
* One label should be associated with a complete time series. In case of multiple labels, often the last label is taken as the label for the complete sequence. 
  Another possibility is to take the majority label.
* In splitting the data into training, validation and test sets, it might be necessary to make sure that sample subject (such as test persons) for which multiple sequences are available, are not present in both train and validation/test set. The same holds for subsequences that originates from the same original sequence (in case of sliding windows).
* The Keras function ``keras.utils.np_utils.to_categorical`` can be used to transform an array of class labels to binary class labels.
* Data doesn't need to be normalized. Every model mcfly produces starts by normalizing data through a Batch Normalization layer. 
  This means that training data is used to learn mean and standard deviation of each channel and timestep.
