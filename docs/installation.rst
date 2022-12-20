Installation
============

Prerequisites:

* Python 3.7, 3.8, 3.9 or 3.10
* pip
* Tensorflow 2

Installing all dependencies in separate conda environment:

.. code:: sh

   conda env create -f environment.yml

   # activate this new environment
   source activate mcfly

To install the package, run in the project directory:

.. code:: sh

   pip install mcfly



Visualization
~~~~~~~~~~~~~

We build a tool to visualize the configuration and performance of the
models. The tool can be found on http://nlesc.github.io/mcfly/. To run
the model visualization on your own computer, cd to the ``html``
directory and start up a python web server:

.. code:: sh

    python -m http.server 8888 &

Navigate to ``http://localhost:8888/`` in your browser to open the
visualization. For a more elaborate description of the visualization see
`user
manual <https://mcfly.readthedocs.io/en/latest/user_manual.html>`__.
