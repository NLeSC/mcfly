import os
from setuptools import setup

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "mcfly",
    version = "0.0.1",
    description = ("Deep learning for time series data"),
    license = "Apache 2.0",
    keywords = "Python",
    url = "https://github.com/NLeSC/mcfly",
    packages=['mcfly'],
    long_description=read('README.md'),
    classifiers=[
        'Development Status :: 1 - Planning',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python',
    ],
)
