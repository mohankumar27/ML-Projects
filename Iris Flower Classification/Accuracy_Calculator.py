# -*- coding: utf-8 -*-
"""
Created on Oct 2018

@author: Mohan Kumar S
"""

"""Accuracy calculator  1=> suitable model 0=> not suitable model"""
def F1_Score(confusion_matrix):
    precision_sum=0
    recall_sum=0
    column_sum = confusion_matrix.sum(axis=0)
    row_sum = confusion_matrix.sum(axis=1)
    for i in range(len(confusion_matrix)):
        precision_sum += confusion_matrix[i][i]/ column_sum[i]
        
    precision_average = precision_sum /len(confusion_matrix)
    
    for i in range(len(confusion_matrix)):
        recall_sum += confusion_matrix[i][i]/ row_sum[i]
        
    recall_average = recall_sum /len(confusion_matrix)
    
    f1_score = 2*((precision_average*recall_average)/(precision_average+recall_average)) #more the value is near to '1', more accurate the model is.    
    return f1_score
    