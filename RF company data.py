# -*- coding: utf-8 -*-
"""
Created on Sun May  2 21:22:50 2021

@author: Shubham
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split 

data = pd.read_csv("D:\\Data Science study\\Assignment of Data Science\\Sent\\13 Decision Tree\\Company_Data.csv")
data.head()
data.describe
data[data.isnull().any(axis=1)]   #There are no null values in the data.
print(data.info())
data.columns
colnames = list(data.columns)
predictors = colnames[1:]
target = colnames[0]

#let us see the boxplot for the given data 
sb.boxplot(data = data)

#There are few plots with the outliers present in them so we will create a function  for finding out these outliers

outliers=[]
def detect_outlier(data_1):         # Creating a function for finding the outliers
    outliers.clear()
    threshold=3
    mean_1 = np.mean(data_1)
    std_1 =np.std(data_1)
    
    
    for y in data_1:
        z_score= (y - mean_1)/std_1 
        if np.abs(z_score) > threshold:
            outliers.append(y)
    return outliers

# Outliers present in the sales
outlier_sales = detect_outlier(data.Sales)
print(outlier_sales)
len(outlier_sales)

# Outliers present in the CompPrice
outlier_compprice = detect_outlier(data.CompPrice)
print(outlier_compprice)
len(outlier_compprice)

# Outliers present in the Price 
outlier_price = detect_outlier(data.Price)
print(outlier_price)
len(outlier_price)

#lets create the function which will take an index number as input and gives out 
#the countplot for that particular column

def count_plot(x):
    plot = sb.countplot(x = data.columns[x], data = data, palette = 'hls')
    return plot


#let us see the plot of all the columns in the data
count_plot(0)
count_plot(1)
count_plot(2)
count_plot(3)
count_plot(4)
count_plot(5)
count_plot(6)
count_plot(7)
count_plot(8)
count_plot(9)
count_plot(10)

#Lets see how a plot against the sales look like
pd.crosstab(data.Sales,data.CompPrice).plot(kind = "bar")
pd.crosstab(data.Sales,data.Advertising).plot(kind = "bar")
pd.crosstab(data.Sales,data.Price).plot(kind = "bar")


#we will create new dataframe with the encoded data since the decision tree classifier can't handle the categorical string data
#first we will create a dictionary for the values which needs to be replaced.

data.dtypes

data["ShelveLoc"].value_counts()

data["Urban"].value_counts()

data["US"].value_counts()

val_replace = {"ShelveLoc": {"Medium":2, "Bad":3, "Good":1}, "Urban": {"Yes":1, "No":0}, "US": {"Yes":1, "No":0}}

x = data.replace(val_replace).copy()

#Also we will type cast the sales column values into the int because the classifier doesn't accepts floating values which it considers as countinuous values

model_data = x.copy()

model_data.dtypes

# Let's split the data into train and test data so we can have different datas for model building and model testing.
from sklearn.model_selection import train_test_split
train,test = train_test_split(model_data,test_size = 0.2)

x_train = train[predictors]
y_train = train[target].astype(int) # We cannot use the continuous data with integers for the confusion matrix.
y_train.dtype
x_test = test[predictors]
y_test = test[target].astype(int) # We cannot use the continuous data with integers for the confusion matrix.
y_test.dtype

rf_model = RandomForestClassifier(n_jobs=3,oob_score=True,n_estimators=15,criterion="entropy")
# n_estimators -> Number of trees ( you can increase for better accuracy)
# n_jobs -> Parallelization of the computing and signifies the number of jobs 
# running parallel for both fit and predict
# oob_score = True means model has done out of box sampling to make predictions

np.shape(model_data)
#(400, 11)

#### Attributes that comes along with RandomForest function
rf_model.fit(x_train,y_train) # Fitting RandomForestClassifier model from sklearn.ensemble 

rf_model.estimators_ # 

rf_model.classes_ # class labels (output)

rf_model.n_classes_ # Number of levels in class labels 

rf_model.n_features_  # Number of input features in model 10 here.

rf_model.n_outputs_ # Number of outputs when fit performed

rf_model.oob_score_  # 0.13125

# We will first see what is the training accuracy of our model

rf_model.predict(x_train)

y_train_pred = rf_model.predict(x_train)

y_train.reset_index(drop = True, inplace = True)

print(y_train)

confusion_matrix(y_train, y_train_pred)

pd.crosstab(y_train, y_train_pred)

accuracy_train = np.mean(y_train == y_train_pred)
accuracy_train

# We will predict the test data we have previously created for testing purpose.

rf_model.predict(x_test)

y_test_pred = rf_model.predict(x_test)

# Let's plot a confusion matrix for the test data

confusion_matrix(y_test, y_test_pred)

# Let's have crosstable for the test data

pd.crosstab(y_test, y_test_pred)

# Let's see the accuracy of the test data

accuracy_test = np.mean(y_test == y_test_pred)

accuracy_test*100
#23.75 %