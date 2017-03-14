# entroPy
Maximum entropy (MaxEnt) classifier in Python with CPython extension.

Note:only test in Python 2.7 and Ubuntu 16.04.


* [wikipedia](https://en.wikipedia.org/wiki/Multinomial_logistic_regression) MaxEnt on Wikipedia

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

me.append(some_feats,label)

ret=me.train(L2=1)

me.saveModel(model_name)

me.loadModel(model_name)

re=me.classify(some_feats)
```
-----
##example:

see:example_titanic.py
* [data](https://www.kaggle.com/c/titanic/data) the competition about Titanic on Kaggle.


trained on train.csv,test on test.csv and gendermodel.csv, the final accuracy is  82.30%.





