# -*- coding: utf-8 -*-
"""
Created on Dec 2018

@author: Mohan Kumar S
"""
# Text Classification for Sentiment Analysis

#%reset -f   
#Importing the Libraries

import re
import nltk
import pickle
from sklearn.datasets import load_files
from nltk.corpus import stopwords
nltk.download('stopwords')

#Importing the dataset

reviews = load_files('txt_sentoken/') #this folder path contains all preprocessed negative and positive reviews that can be used for training the data
X,y = reviews.data,reviews.target #X-> contains all reviews, y-> contains 0 or 1 representing negative or positive

#Storing as pickle files

# loading large sets of data would be time consuming whenever the code is run.
# Hence once the data is loaded, storing them as pickle files would reduce the time consuming process of
# loading the files again as these pickel files can be used instead
with open('X.pickle','wb') as f:
    pickle.dump(X,f)
    
with open('y.pickle','wb') as f:
    pickle.dump(y,f)
    
# once X and y are loaded and stored as pickle files, run the below lines in future
# to load them quickly instead of running the above 'Importing the dataset' code
'''   
# Loading pickle files
with open('X.pickle','rb') as f:
    X = pickle.load(f)
    
with open('y.pickle','rb') as f:
    y = pickle.load(f)
'''

#Preprocessing the data

preprocessed_review=[]
for i in range(len(X)):
    review = re.sub(r'\W',' ',str(X[i])) # removes all puntuation characters in each review
    review = review.lower() # converts review to lowercase
    review = re.sub(r'\s+[a-z]\s+',' ',review) #replaces single letter characters occuring in middle of the sentences with a space
    review = re.sub(r'^[a-z]\s+',' ',review) #replaces single letter characters occuring at the beginning with a space
    review = re.sub(r'\s+',' ',review) #replaces multiple spaces in a sentence with a single space
    preprocessed_review.append(review)

# below commented lines are for visuvalising how Bag of words model and TfIdf model works individually
# the 'TfidfVectorizer' model does both the job of Bag of words model and TfIdf model
# so the below commented lines need not be run..
'''
#Bag of words model

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(max_features=2000, min_df=3, max_df=0.6, stop_words=stopwords.words('english')) #max_features-> no/: of most frequent words to be considered
#min_df -> excluding words that occurs in 3 reviews or less, max_df -> excluding words that occur in more than 60% of the reviews
# stop_words -> excluding stop words
X = vectorizer.fit_transform(preprocessed_review).toarray()

#TfIdf model -> converts the above bag of words model to TF-IDF model

from sklearn.feature_extraction.text import TfidfTransformer
transformer = TfidfTransformer()
X = transformer.fit_transform(X).toarray()
'''
# TfIdfVectorizer -> does the job of both bag of words model and tfidf model

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=2000, min_df=10, max_df=0.6, stop_words=stopwords.words('english')) #max_features-> no/: of most frequent words to be considered
#min_df -> excluding words that occurs in 3 reviews or less, max_df -> excluding words that occur in more than 60% of the reviews
# stop_words -> excluding stop words
X = vectorizer.fit_transform(preprocessed_review).toarray()
    
# splitting the dataset into train and test data

from sklearn.model_selection import train_test_split
review_train, review_test, type_train, type_test = train_test_split(X, y, test_size=0.2, random_state=0 ) # 80% of data for training and remaining 20% for testing purposes

#Logistic regression classification

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(review_train,type_train)

# Testing the model

type_pred = classifier.predict(review_test) #Logistic Prediction

#Making Confusion matrix

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(type_test,type_pred)

# pickle the 'classifer' variable to be used later for Twitter sentiment analysis purpose

with open('sentiment_classifer.pickle','wb') as f:
    pickle.dump(classifier,f)
 
# pickle the above variable 'vectorizer' to be used later for Twitter sentiment analysis purpose

with open('tfidfmodel.pickle','wb') as f: #since we have trained our classifer model based on this vectorizer variable, it is necessary to store this variable for future sentiment analysis purposes
    pickle.dump(vectorizer,f)

# importing and testing for user inputs

with open('tfidfmodel.pickle','rb') as f:
    tf_idf_model = pickle.load(f)

with open('sentiment_classifer.pickle','rb') as f:
    clf = pickle.load(f)

sample_review = ["Terrible cartoon. neither educational nor entertaining"]
sample_review = tf_idf_model.transform(sample_review).toarray()
print(clf.predict(sample_review)) # prints 0 or 1 corresponding to negative or positive review
