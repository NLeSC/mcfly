<p align="left">
  <img src="mcflylogo.png" width="200"/>
</p>

[![Build Status](https://travis-ci.org/NLeSC/mcfly.svg?branch=master)](https://travis-ci.org/NLeSC/mcfly)
[![Code quality](https://scrutinizer-ci.com/g/NLeSC/mcfly/badges/quality-score.png?b=master)](https://scrutinizer-ci.com/g/NLeSC/mcfly/)
[![Coverage](https://scrutinizer-ci.com/g/NLeSC/mcfly/badges/coverage.png?b=master)](https://scrutinizer-ci.com/g/NLeSC/mcfly/statistics/)
<a href="https://zenhub.io"><img src="https://raw.githubusercontent.com/ZenHubIO/support/master/zenhub-badge.png"></a>
[![DOI](https://zenodo.org/badge/59207352.svg)](https://zenodo.org/badge/latestdoi/59207352)
[![Binder](http://mybinder.org/badge.svg)](http://mybinder.org:/repo/nlesc/mcfly)

The goal of mcfly is to ease using deep learning technology for time series classification. The advantages of deep learning algorithms is that it can handle raw data directly with no need to compute signal features, it does not require a expert domain knowledge about the data, and it has been shown to be competitive with conventional machine learning techniques. As an example, you can apply mcfly on ,accelerometer data for activity classification, as shown in [the tutorial](https://github.com/NLeSC/mcfly/tree/master/notebooks/tutorial).

## Installation
Prerequisites:
- Python 2.7 or 3.5 (The tutorial is only tested in Python 3.5)
- pip

To install the package, run in the project directory:

`pip install .`

### Installing on Windows
When installing on Windows, there are a few things to take into consideration. The preferred (in other words: easiest) way to install Keras and mcfly is as follows:
* Use [Anaconda](https://www.continuum.io/downloads)
* Install numpy and scipy through the conda package manager (and not with pip)
* Use TensorFlow, and not Theano as a backend (this is the default setting in Keras)
* Preferably use Python version >= 3.5, because Tensorflow is available on pypi for python 3.5 only
* To install mcfly, run `pip install .` in the cmd prompt.

## Visualization
We build a tool to visualize the configuration and performance of the models. To run the  model visualization, cd to the `html` directory and start up a python web server:

`python -m http.server 8888 &`

Navigate to `http://localhost:8888/` in your browser to open the visualization. For a more elaborate description of the visualization see [user manual](https://github.com/NLeSC/mcfly/wiki/User-manual).


## User documentation
* [Wiki page](https://github.com/NLeSC/mcfly/wiki/Home---mcfly)
* [User manual](https://github.com/NLeSC/mcfly/wiki/User-manual)
* [Code documentation](http://mcfly.readthedocs.io/en/latest/)
* [Technical Documentation](https://github.com/NLeSC/mcfly/wiki/Technical-documentation)
* [Information on Data preprocessing](https://github.com/NLeSC/mcfly/wiki/Data-preprocessing)

## Contributing
You are welcome to contribute to the code via pull requests. Please have a look at the [NLeSC guide](https://nlesc.gitbooks.io/guide/content/software/software_overview.html) for guidelines about software development.

We use numpy-style docstrings for code documentation.

## Licensing
Source code and data of mcfly are licensed under the Apache License, version 2.0.
