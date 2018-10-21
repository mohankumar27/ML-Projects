# -*- coding: utf-8 -*-
"""
Created on Oct  2018

@author: Mohan Kumar S
"""

""" Data preparation and splitting training data and test data"""
import pandas as pd #pandas is imported to read csv files
iris_dataset = pd.read_csv("iris.csv") #reading the dataset. Load your dataset here..
independent_variables = iris_dataset.iloc[:,1:5].values #independent variables include sepalLength, sepalWidth, petalLength, petalWidth
dependent_variable = iris_dataset.iloc[:,5].values #dependent variables include plant species ie setosa, versicolor, verginica

from sklearn.preprocessing import LabelEncoder #LabelEncoder is used to convert categorical variables ie coloum containing names into numerical values for futhur processing
labelencoder_y  = LabelEncoder()
dependent_variable = labelencoder_y.fit_transform(dependent_variable)  #converts names setosa, versicolor and verginica into 1,2 and 3 respectively

from sklearn.cross_validation import train_test_split #splitting the dataset into testing and training data
independent_variables_train,independent_variables_test, dependent_variable_train,\
    dependent_variable_test = train_test_split(independent_variables,\
                                                dependent_variable,test_size=0.2,random_state=0) #'\' is used to break lines of code instead of one single long line
    
"""Logistic Regression Classification"""
from sklearn.linear_model import LogisticRegression
classifier_LR = LogisticRegression(random_state=0) # random_state is used to obtain similar results in future
classifier_LR.fit(independent_variables_train,dependent_variable_train) #Training the model with training dataset

#Predicting the test set results
dependent_variable_predicted_LR = classifier_LR.predict(independent_variables_test) #Predicting the values of test set from the trained model

#Making Confusion matrix
from sklearn.metrics import confusion_matrix
cm_LR = confusion_matrix(dependent_variable_test,dependent_variable_predicted_LR) #used to compare predicted and actual values and provide a matrix revealing a total number of accurately predicted and false values


"""KNN Classification"""
from sklearn.neighbors import KNeighborsClassifier
classifier_KNN = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2) #n_neighbors=>number of neighbours to take into account, metric,p => used to ensure euclidean distance to be used for measuring distance between points
classifier_KNN.fit(independent_variables_train,dependent_variable_train)

#Predicting the test set results
dependent_variable_predicted_KNN = classifier_KNN.predict(independent_variables_test)

#Making Confusion matrix
from sklearn.metrics import confusion_matrix
cm_KNN = confusion_matrix(dependent_variable_test,dependent_variable_predicted_KNN)


"""SVM Classification"""
from sklearn.svm import SVC
classifier_SVM = SVC(kernel='rbf', random_state=0) # kernal=> kernal type to be used for the algorithm
classifier_SVM.fit(independent_variables_train,dependent_variable_train)

#Predicting the test set results
dependent_variable_predicted_SVM = classifier_SVM.predict(independent_variables_test)

#Making Confusion matrix
from sklearn.metrics import confusion_matrix
cm_SVM = confusion_matrix(dependent_variable_test,dependent_variable_predicted_SVM)


"""Naive_Bayes Classification"""
from sklearn.naive_bayes import GaussianNB
classifier_NB = GaussianNB()
classifier_NB.fit(independent_variables_train,dependent_variable_train)

#Predicting the test set results
dependent_variable_predicted_NB = classifier_NB.predict(independent_variables_test)

#Making Confusion matrix
from sklearn.metrics import confusion_matrix
cm_NB = confusion_matrix(dependent_variable_test,dependent_variable_predicted_NB)


"""Accuracy Calculation"""
from Accuracy_Calculator import F1_Score #For Logic, refer Accuracy_Calculator.py file
accuracy_LR = F1_Score(cm_LR)
accuracy_KNN = F1_Score(cm_KNN)
accuracy_SVM = F1_Score(cm_SVM)
accuracy_NB = F1_Score(cm_NB)