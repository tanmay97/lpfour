import matplotlib.pyplot as plt
import pandas as pd

# Read Dataset
dataset=pd.read_csv("hours.csv")
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1].values

# Import the Linear Regression and Create object of it
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X,y)
Accuracy=regressor.score(X, y)*100
print("Accuracy :")
print(Accuracy)

# Predict the value using Regressor Object
y_pred=regressor.predict([[10]])
print(y_pred)

# Take user input
hours=int(input('Enter the no of hours '))

#calculate the value of y
eq=regressor.coef_*hours+regressor.intercept_
y1='%f*%f+%f' %(regressor.coef_,hours,regressor.intercept_)
print("y :")
print(y1)
print("Risk Score : ", eq[0])
# print(X)
# print(y)
plt.plot(X,y,'o')
plt.plot(X,regressor.predict(X));
plt.show()
# %matplotlib notebook


# lr.py
# dc.py
# knn.py
# km.py
# svm.py
# pca.py
# des.py
# aes.py
# dh.py
# rsa.java
# lpfour.zip