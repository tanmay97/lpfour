import pandas as pd
import numpy as np
data = pd.read_csv('svm_data.csv')
data.head()
data.shape
data.count()
x=data.iloc[:,1:]
y=data.iloc[:,0]

x.shape
len(y)
type(x)
type(y)

x=data.iloc[:,1:].values
y=data.iloc[:,0].values
type(x)
type(y)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.25)

#testsize
#test size by degault 0.25 
#should be between 0 and 1
#represents the absolute number of test samples

#trainsize
#if not specified complemet to the testsize 
#between 0 and 1
x_train.shape
x_test.shape
y_train.shape
y_test.shape

from sklearn.svm import SVC
classifier = SVC(kernel='linear')
# C= support vector classification
# kernel 
#default = rbf
#{rbf,linear,poly,sigmoid,precomputed,callable}

classifier.fit(x_train, y_train)
#Fit the SVM model according to the given training data.
y_pred=classifier.predict(x_test)
#Perform classification on samples in x_test.
result=pd.DataFrame({'Actual':y_test,'Predicted':y_pred})
result.head()

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,y_pred)

from sklearn.metrics import classification_report
classification_report(y_test, y_pred)

from sklearn.metrics import accuracy_score
print (accuracy_score(y_test, y_pred))