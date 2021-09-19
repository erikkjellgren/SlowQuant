[![Build Status](https://travis-ci.com/erikkjellgren/SlowQuant.svg?branch=master)](https://travis-ci.com/erikkjellgren/SlowQuant)
[![Coverage Status](https://coveralls.io/repos/github/erikkjellgren/SlowQuant/badge.svg?branch=master)](https://coveralls.io/github/erikkjellgren/SlowQuant?branch=master)
[![Documentation Status](https://readthedocs.org/projects/slowquant/badge/?version=latest)](http://slowquant.readthedocs.io/en/latest/?badge=latest)

# SlowQuant

![SlowQuant logo](https://cloud.githubusercontent.com/assets/11976167/26658726/5e125b02-466c-11e7-8790-8412789fc9fb.jpg)

SlowQuant is a molecular quantum chemistry program written in python. Even the computational demanding parts are written in python, so it lacks speed, thus the name SlowQuant.

Documentation can be found at:

http://slowquant.readthedocs.io/en/latest/

The program is run by:

```
python SlowQuant.py MOLECULE SETTINGS
```
  
As a ready to run example:

```
python SlowQuant.py H2O.csv settingExample.csv
```
  
# Setup

The program is setup by executing the following command. See documentation for more detailed installation.

```
python setup.py build_ext --inplace
```


# Requirements

- cython 0.29.24
- gcc 9.3.0
- Python 3.9.7
- numba 0.53.1
- numpy 1.20.3
- Python 3.9.7
- scipy 1.7.1
