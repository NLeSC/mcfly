Installation
============

Prerequisites:

* Python 2.7, 3.5 or 3.6
* pip

Installing all dependencies in sparate conda environment:

``conda env create -f environment.yml``

``source activate mcfly``

To install the package, run in the project directory:

``pip install .``

Installing on Windows
---------------------
When installing on Windows, there are a few things to take into consideration. The preferred (in other words: easiest) way to install Keras and mcfly is as follows:

* Use `Anaconda <https://www.continuum.io/downloads>`_
* Use Python 3.x, because tensorflow is not available on Windows for Python 2.7
* Install numpy and scipy through the conda package manager (and not with pip)
* To install mcfly, run `pip install mcfly` in the cmd prompt.
* Loading and saving models can give problems on Windows, see https://github.com/NLeSC/mcfly-tutorial/issues/17

Visualization
---------------------
We build a tool to visualize the configuration and performance of the models. The tool can be found on http://nlesc.github.io/mcfly/. To run the  model visualization on your own computer, cd to the `html` directory and start up a python web server:

``python -m http.server 8888 &``

Navigate to `http://localhost:8888/` in your browser to open the visualization. 
For a more elaborate description of the visualization see `user manual <https://github.com/NLeSC/mcfly/wiki/User-manual>`_.
