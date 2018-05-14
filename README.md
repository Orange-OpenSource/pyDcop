# Dcop python

[![Documentation Status](https://readthedocs.org/projects/pydcop/badge/?version=latest)](http://pydcop.readthedocs.io/en/latest/?badge=latest)
[![Build Status](https://travis-ci.org/Orange-OpenSource/pyDcop.svg?branch=master)](https://travis-ci.org/Orange-OpenSource/pyDcop)

pyDCOP is a python library for Distributed Constraints Optimization.
It contains implementations of several standard DCOP algorithms (MaxSum, DSA,
DPOP, etc.) and allows you to develop your own algorithms.

pyDCOP runs on python >= 3.5.
 
## Installation

Using pip is recommended, on ubuntu :

    sudo apt-get install python3-setuptools
    sudo apt-get install python3-pip


I also recommend installing pyDCOP in a virtualenv, in order to avoid any
conflict with other applications you might have:

     python3 -m venv ~/.pydcop
     source ~/.pydcop/bin/activate

For now, installation is only from source :

    cd pydcop
    pip install .

Or without pip, simply use :

    python3 setup.py install
    
When developing on DCOP-python, one would rather use the following command,
which installs pydcop in development mode and test dependencies:

    pip install -e .[test]

To generate documentation, you need to install the corresponding dependencies:

    pip install -e .[doc]

