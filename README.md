# entroPy
Maximum entropy (MaxEnt) classifier in Python with CPython extension.

Note:only test in Python 2.7 and Ubuntu 16.04.


* [wikipedia](https://en.wikipedia.org/wiki/Multinomial_logistic_regression)

-----

[![License](https://img.shields.io/badge/license-GPL3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0.en.html)
-----


## Installation
for CPython,you must install python-dev on Linux System,such as Ubuntu.

To install entroPy, `cd` to the entroPy folder and run the install command:
```sh
sudo python setup.py install
```
-----
##tutorial:
```sh
from entroPy  import MaxEntClassify

me=MaxEntClassify()

me.append(feats,label)

ret=me.train(L2=1)

me.saveModel(u'titanic')

me.loadModel(u'titanic')

re=me.classify(feats)
```
-----




