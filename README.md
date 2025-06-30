<p align="left">
  <img src="https://raw.githubusercontent.com/NLeSC/mcfly/master/mcflylogo_with_regression.png" width="200"/>
</p>

[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/NLeSC/mcfly/CI_build.yml?branch=main)](https://github.com/NLeSC/mcfly/actions/workflows/CI_build.yml)
[![Coverage](https://scrutinizer-ci.com/g/NLeSC/mcfly/badges/coverage.png?b=master)](https://scrutinizer-ci.com/g/NLeSC/mcfly/statistics/)
[![PyPI](https://img.shields.io/pypi/v/mcfly.svg)](https://pypi.python.org/pypi/mcfly/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.596127.svg)](https://doi.org/10.5281/zenodo.596127)
[![Binder](http://mybinder.org/badge.svg)](http://mybinder.org:/repo/nlesc/mcfly)
<!-- The first 12 lines are skipped while generating 'long description' (see setup.py)) -->

The goal of mcfly is to ease the use of deep learning technology for time series classification and regression. The advantage of deep learning is that it can handle raw data directly, without the need to compute signal features. Deep learning does not require  expert domain knowledge about the data, and has been shown to be competitive with conventional machine learning techniques. As an example, you can apply mcfly on accelerometer data for activity classification, as shown in [the tutorial](https://github.com/NLeSC/mcfly-tutorial).

If you use mcfly in your research, please cite the following software paper:

D. van Kuppevelt, C. Meijer, F. Huber, A. van der Ploeg, S. Georgievska, V.T. van Hees. _Mcfly: Automated deep learning on time series._
SoftwareX,
Volume 12,
2020.
[doi: 10.1016/j.softx.2020.100548](https://doi.org/10.1016/j.softx.2020.100548)

## Installation
Prerequisites:
- Python 3.10, 3.11
- pip
- Tensorflow 2, PyTorch or JAX

Installing all dependencies in separate conda environment:
```sh
conda env create -f environment.yml

# activate this new environment
source activate mcfly
```

To install the package, run one of the following commands in the project directory:

- `pip install mcfly[tensorflow]`
- `pip install mcfly[torch]`
- `pip install mcfly[jax]`

Please note: If you are not using tensorflow, you have to set the environment variable `KERAS_BACKEND` accordingly to your chosen backend.

For GPU support take a look at the latest version of the requirements section "[most stable GPU environment](https://keras.io/getting_started/#most-stable-gpu-environment)" inside the Keras documentation or directly in their [GitHub repository](https://github.com/keras-team/keras).

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
