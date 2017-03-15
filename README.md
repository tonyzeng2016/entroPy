# entroPy
Maximum entropy (MaxEnt) classifier in Python with CPython extension.

Note:only test in Python 2.7 and on Ubuntu 16.04.


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
## tutorial and API:
```sh
from entroPy  import MaxEntClassify

me=MaxEntClassify() 

me.append(single_data_feats,single_data_label)

ret=me.train(L2=1)

me.saveModel(model_name)

me.loadModel(model_name)

re=me.classify(single_data_feats)
```
-----
## example:

[see:](https://github.com/tonyzeng2016/entroPy/blob/master/example_titanic.py) example_titanic.py

[data:](https://www.kaggle.com/c/titanic/data) Titanic on Kaggle.

after prepared the data,to run the example:
```sh
python example_titanic.py
```
trained on train.csv,test on test.csv and gendermodel.csv, the final accuracy is  82.30%(features chosen arbitrarily).





