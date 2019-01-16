<p align="left">
  <img src="https://github.com/NLeSC/mcfly/blob/master/mcflylogo.png" width="200"/>
</p>

[![Build Status](https://travis-ci.org/NLeSC/mcfly.svg?branch=master)](https://travis-ci.org/NLeSC/mcfly)
[![AppVeyor Build Status](https://ci.appveyor.com/api/projects/status/lv8hih1hvxbuu5f7/branch/master?svg=true)](https://ci.appveyor.com/project/NLeSC/mcfly/)
[![Code quality](https://scrutinizer-ci.com/g/NLeSC/mcfly/badges/quality-score.png?b=master)](https://scrutinizer-ci.com/g/NLeSC/mcfly/)
[![Coverage](https://scrutinizer-ci.com/g/NLeSC/mcfly/badges/coverage.png?b=master)](https://scrutinizer-ci.com/g/NLeSC/mcfly/statistics/)
[![PyPI](https://img.shields.io/pypi/v/mcfly.svg)](https://pypi.python.org/pypi/mcfly/)
[![DOI](https://zenodo.org/badge/59207352.svg)](https://zenodo.org/badge/latestdoi/59207352)
[![Binder](http://mybinder.org/badge.svg)](http://mybinder.org:/repo/nlesc/mcfly)
<!-- The first 12 lines are skipped while generating 'long description' (see setup.py)) -->

The goal of mcfly is to ease the use of deep learning technology for time series classification. The advantage of deep learning is that it can handle raw data directly, without the need to compute signal features. Deep learning does not require  expert domain knowledge about the data, and has been shown to be competitive with conventional machine learning techniques. As an example, you can apply mcfly on accelerometer data for activity classification, as shown in [the tutorial](https://github.com/NLeSC/mcfly-tutorial).

## Installation
Prerequisites:
- Python 2.7, 3.5 or 3.6
- pip

Installing all dependencies in sparate conda environment:
```sh
conda env create -f environment.yml

# activate this new environment
source activate mcfly
```

To install the package, run in the project directory:

`pip install .`

### Installing on Windows
When installing on Windows, there are a few things to take into consideration. The preferred (in other words: easiest) way to install Keras and mcfly is as follows:
* Use [Anaconda](https://www.continuum.io/downloads)
* Use Python 3.x, because tensorflow is not available on Windows for Python 2.7
* Install numpy and scipy through the conda package manager (and not with pip)
* To install mcfly, run `pip install mcfly` in the cmd prompt.
* Loading and saving models can give problems on Windows, see https://github.com/NLeSC/mcfly-tutorial/issues/17

## Visualization
We build a tool to visualize the configuration and performance of the models. The tool can be found on http://nlesc.github.io/mcfly/. To run the  model visualization on your own computer, cd to the `html` directory and start up a python web server:

`python -m http.server 8888 &`

Navigate to `http://localhost:8888/` in your browser to open the visualization. For a more elaborate description of the visualization see [user manual](https://mcfly.readthedocs.io/en/latest/user_manual.html).


## User documentation
[User and code documentation](https://mcfly.readthedocs.io).

## Contributing
You are welcome to contribute to the code via pull requests. Please have a look at the [NLeSC guide](https://nlesc.gitbooks.io/guide/content/software/software_overview.html) for guidelines about software development.

We use numpy-style docstrings for code documentation.

## Licensing
Source code and data of mcfly are licensed under the Apache License, version 2.0.
