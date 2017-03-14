#encoding=utf-8
'''
Created on 2015-11-12

@author: Administrator
'''
from __future__ import division
from entroPy  import MaxEntClassify
import csv
def train(datafile):
    me=MaxEntClassify()
    with open(datafile) as fh:
        data=csv.reader(fh);
        for index_,row in enumerate(data):
            #print('{0} {1}'.format(index_,row[1:]));
            if index_<=0:continue
            passenger=Passenger(*row[1:])
            feats=passenger.feature();
            #print('{0} {1}'.format(index_,passenger.embarked))
            u'''
            append single data as: ['a','b','c'] 'label'
            Python:['a','b','c'], 'label'
            '''
            me.append(feats[1:],feats[0])
            #print(passerger.feature())
    u'''train model with L2 param
    train result:
    0--training process finished normally ;
    non-0--some error occur.
    '''
    ret=me.train(L2=1);
    print(u'train result:{0}'.format(ret))
    u'''save the model'''
    me.saveModel(u'titanic')
    #print(ret)
def test(testFile,refenceFile):
    me=MaxEntClassify()
    u'''load model'''
    me.loadModel(u'titanic')
    feats=None
    re=[]
    re2=[]
    with open(testFile) as fh:
        data=csv.reader(fh);
        for index_,row in enumerate(data):
            if index_<=0:continue
            #passerger=Passerger(None,row[1],row[2],row[3],row[4],row[5],row[6],row[7],row[8],row[9],row[10])
            passerger=Passenger(None,*row[1:])
            feats=passerger.feature()
            re_=me.classify(feats[1:])
            re.append(sorted(re_.items(),key=lambda a:a[1],reverse=True)[0][0])
    #print(re)
    with open(refenceFile) as fh:
        data=csv.reader(fh);
        for index_,row in enumerate(data):
            if index_<=0:continue
            re2.append(row[1]);
    
    re2=['No' if item=='0' else 'Yes' for item in re2]
    #print(re2)
    n=len(re2)
    error=sum([1 if re[i]!=re2[i] else 0 for i in range(n)])
    print   u'accuracy %f' %(1-error/n)

    #return re;
            
def predict():
    u'''load the model and predict'''
    me=MaxEntClassify()
    me.loadModel(u'titanic')
    u'''the predict result is dict object,category as key,prob as value'''
    print(me.classify([u'3rd', u'male', u'age-0', u'C', u'sibsp-0', u'parch-2']))
    pass#['No', '3rd', 'female', 'age-0', 'S', 'sibsp-1', 'parch-2']
class Passenger:
    u'''wrapper every passenger'''
    def __init__(self,survival,pclass,name,sex,age,sibsp,parch,ticket,fare,cabin,embarked):
        self.pclassdict={u'1':u'1st',u'2':u'2nd',u'3':u'3rd'}
        self.survival=u'No' if survival==u'0' else u'Yes'
        self.pclass=self.pclassdict[pclass]
        self.name=name              #Name
        self.sex=sex                #Sex
        self.age=float(age) if  len(age)>0 else -1               #Age
        self.sibsp=sibsp#float(sibsp) if  len(sibsp)>0 else 0           #Number of Siblings/Spouses Aboard     兄弟/夫妻
        self.parch=parch#float(parch) if  len(parch)>0 else 0            #Number of Parents/Children Aboard     父母/子女
        self.ticket=ticket          #Ticket Number                船票编号
        self.fare=float(fare) if  len(fare)>0 else 0              #Passenger Fare                船票价格
        self.cabin=cabin            #Cabin                    客舱号码
        self.embarked=embarked      #Port of Embarkation            登船点
    def feature(self):
        if(0<self.age<16):
            ageStr=1;
        elif(16<=self.age<30):
            ageStr=2;
        elif(30<=self.age<50):
            ageStr=3;
        elif(50<=self.age):
            ageStr=4;
        else:
            ageStr=0;
        feat=[]
        feat.append(self.survival);
 
        feat.append(u'class-{0}'.format(self.pclass));
 
        #feat.append(self.name)
   
        feat.append(self.sex)
  
        feat.append(u'age-{0}'.format(ageStr))

        feat.append(u'sibsp-{0}'.format(self.sibsp))
    
        #feat.append(u'parch-{0}'.format(self.parch))

        feat.append(u'ticket-{0}'.format(self.ticket))
        feat.append(u'fare-{0}'.format(self.fare))
 
        #feat.append(u'carbin-{0}'.format(self.cabin))
        #feat.append(u'embarked-{0}'.format(self.embarked))
        return feat
if __name__==u'__main__':
    trainFile=u'train.csv'
    testFile=u'test.csv'
    testFile1=u'gendermodel.csv'
    train(trainFile);
    test(testFile,testFile1);
    predict()
