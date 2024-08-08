import os
from setuptools import setup, find_packages

with open(os.path.join(os.path.dirname(__file__), 'mcfly/_version.py')) as versionpy:
    exec(versionpy.read())

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "mcfly",
    version = __version__,
    description = ("Deep learning for time series data"),
    license = "Apache Software License 2.0",
    keywords = "Python",
    url = "https://github.com/NLeSC/mcfly",
    packages = find_packages(),
    long_description=read('README.md'),
    long_description_content_type='text/markdown',
    classifiers=[
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python',
    ],
    test_suite="tests",
    python_requires='>=3.10',
    install_requires=[
        "numpy",
        "scikit-learn",
        "scipy",
        "tensorflow>=2.0.0",
        "h5py",
    ],
    extras_require={"dev": ["coverage",
                            "prospector[with_pyroma]==1.7.7",
                            "pytest",
                            "pytest-cov",
                            ],
                    "publishing": [
                        "build",
                        "twine",
                        "wheel",
                        "sphinx",
                        "sphinx-rtd-theme"
                    ]
    }
)
