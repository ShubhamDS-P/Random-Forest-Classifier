# -*- coding: utf-8 -*-
"""
Created on Mon May  3 13:20:47 2021

@author: Shubham
"""

import pandas as pd
import numpy as np

# Importing the data into the environment

fraud_data = pd.read_csv("D:\\Data Science study\\Assignment of Data Science\\Sent\\13 Decision Tree\\Fraud_check.csv")
fraud_data
fraud_data.head()
fraud_data.info()

# Checking the null values in the data

null = fraud_data.isnull().any(axis = 1)
null.describe()

# This shows there are no null values present in the dataframe

fraud_data.columns

# Changing the column names, since the dots in them creates errors while running the code.

fraud_data.rename(columns={'Marital.Status':'Marital_Status', 'Taxable.Income':'Taxable_Income', 'City.Population':'City_Population', 'Work.Experience':'Work_Experience'},inplace= True)

# Now we will insert a column at the 0th position as an out put column where we will divide our taxable income into 'good' and 'risky'.

fraud_data.insert(0,"output",str) # Since we are going to insert the strings in the column hence we set it as 'str'

# We will modify the output column so it will have the values which we are going to use for the output 'y' as per the 
#requirments mentioned in the problem statement.

fraud_data.loc[fraud_data.Taxable_Income<=30000,"output"] = "Risky"

fraud_data.loc[fraud_data.Taxable_Income>30000,"output"] = "Good"

fraud_data['output'].describe()

type(fraud_data.output)

# Checking the null values once again in the data.

#null = fraud_data.isnull().any(axis = 1)
#null.describe()

# Another method to find the null values in the data.

fraud_data[fraud_data.isnull().any(axis = 1)]

# There are no null values in the data.

print(fraud_data.info())

# Let's see how the data looks in the graphical visuals which we are going to plot next
import seaborn as sb

sb.boxplot(fraud_data.City_Population)

sb.boxplot(fraud_data.Work_Experience)

# Basically there are no outliers in the numerical data 
# Now we will plot some graphs for the remaining data and see how it looks.

fraud_data.columns

graph1 = sb.countplot(x = 'Undergrad', data = fraud_data,palette = 'hls')

graph2 = sb.countplot(x = 'Marital_Status', data = fraud_data, palette = 'hls')

graph2 = sb.countplot(x = 'Urban', data = fraud_data, palette = 'hls')

# we can also have a bar plot against the output

pd.crosstab(fraud_data.Marital_Status,fraud_data.output).plot(kind = 'bar')

pd.crosstab(fraud_data.output,fraud_data.Undergrad).plot(kind = 'bar')

pd.crosstab(fraud_data.Urban,fraud_data.output).plot(kind = 'bar')

# These are some of the visuals we got from the graph and we will build our model next based on the decision tree classifier.
# Before the model building we first have to make all the data in to the proper format for the algorithm 
# Since, as per my knowledge the algorithm can't handle the categorical data so we will assign the numerical values to the categories in each column.
# We can also create dummy variables, but in this case I prefer to do it manually since it is much more easy for me.

fraud_data["Undergrad"].value_counts()

fraud_data["Marital_Status"].value_counts()

fraud_data["Urban"].value_counts()

fraud_data["output"].value_counts()

val_replace = {"Undergrad" : {"YES":1,"NO":0}, "Marital_Status" : {"Single":1, "Married":2, "Divorced":3},
               "Urban" : {"YES":1,"NO":0}, "output" : {"Good": 1, "Risky": 0}}

# We are creating a new dataframe with the name 'x' which will have modified column names of the original dataframe

x = fraud_data.replace(val_replace).copy()

x.drop("Taxable_Income",axis = 1, inplace = True) # We will remove the taxable income column which is of no more use to us in the further process.

#Defining the targets and predictors for the model  building.

colnames = list(x.columns)

predictors = colnames[1:]

target = colnames[0]

from sklearn.model_selection import train_test_split

train,test = train_test_split(x,test_size = 0.3)

# Defining the training and testing predictors and target

x_train = train[predictors]
y_train = train[target]
x_test = test[predictors]
y_test = test[target]

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_jobs=3,oob_score=True,n_estimators=70,criterion="entropy")
# n_estimators -> Number of trees ( you can increase for better accuracy)
# n_jobs -> Parallelization of the computing and signifies the number of jobs 
# running parallel for both fit and predict
# oob_score = True means model has done out of box sampling to make predictions

np.shape(fraud_data)
# (600, 7)

#### Attributes that comes along with RandomForest function
rf_model.fit(x_train,y_train) # Fitting RandomForestClassifier model from sklearn.ensemble 

rf_model.estimators_ # 

rf_model.classes_ # class labels (output)

rf_model.n_classes_ # Number of levels in class labels 

rf_model.n_features_  # Number of input features in model 5 here.

rf_model.n_outputs_ # Number of outputs when fit performed

rf_model.oob_score_  # 0.7428571428571429

# We will predict the test data we have previously created for testing purpose.

rf_model.predict(x_test) # Predicting the values

y_test_pred = rf_model.predict(x_test) # Creating a new dataframe with only predicted values in it.

# Let's plot a confusion matrix for the test data
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, y_test_pred)

# Let's have crosstable for the test data

pd.crosstab(y_test, y_test_pred)

# Let's see the accuracy of the test data

accuracy_test = np.mean(y_test == y_test_pred)

accuracy_test*100
# 73.33333333333333 %

# Let's try another method for the accuracy based on the crosstable and confusion matrix

accuracy = (5+127)/(33+15+5+127)*100
accuracy
# 73.33333333333333 %

# Here both the accuracies are same, so we can conclude that the final accuracy of our model based on the given
# data will be approximately 70%

# According to my thoughts the data given for the process is not enough for the model to reach higher accuracy level
# and due to this limitation we can see that the accuracy drops considerably in the above model.