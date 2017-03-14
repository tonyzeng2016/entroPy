# -*- coding: utf-8 -*-
#
# Copyright (C)  entroPy project
# Author: xh.along <zengxiaolong2015@163.com>
# URL: <under construction>
# For license information, see LICENSE.TXT
#Created on 2015-11-12

class EntroPyException(Exception):pass;
class MaxentNoDataException(EntroPyException):pass;
class MaxentDataTypeException(EntroPyException):pass;

class MaxEntClassify:
    def __init__(self):
        from ._model import MaxEntTrainer                
        self._maxent =MaxEntTrainer()
    def append(self,singleData,label):
        if singleData is None or label is None:return
        if isinstance(singleData,list)==False:
            raise MaxentDataTypeException('MaxEnt: data type error')
        if not isinstance(label,str) and not isinstance(label,unicode):
            raise MaxentDataTypeException('MaxEnt: data type error')
        self._maxent.append(singleData,label);
    def train(self,L2=1.0):
        if self._maxent.exist_data()==False:
            raise MaxentNoDataException('MaxEnt: not data error')
        ret=self._maxent.train(L2);
        return ret;
    def saveModel(self,filename=None):
        if filename is None :return
        if not isinstance(filename,str) and not isinstance(filename,unicode):
            raise MaxentDataTypeException('MaxEnt: data type error')
        if self._maxent.exist_model()==False:
            raise MaxentNoDataException('MaxEnt: not model error')
        ret=self._maxent.save(filename)
        return ret;
    def loadModel(self,filename):
        if filename is None :return
        if not isinstance(filename,str) and not isinstance(filename,unicode):
            raise MaxentDataTypeException('MaxEnt: data type error')
        ret=self._maxent.load(filename)
        return ret;    
    def classify(self,single):
        if single is None :return
        if isinstance(single,list)==False:
            raise MaxentDataTypeException('MaxEnt: data type error')
        if self._maxent.exist_model()==False:
            raise MaxentNoDataException('MaxEnt: not model error')
        return self._maxent.predict(single)

        
__all__=['MaxEntClassify']
