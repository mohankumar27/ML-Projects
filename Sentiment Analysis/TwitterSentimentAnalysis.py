# -*- coding: utf-8 -*-
"""
Created on Dec 2018

@author: Mohan Kumar S
"""
# Twitter Sentiment Analysis

import tweepy # used for authentication and fetching tweets from twitter
import re
import pickle

from tweepy import OAuthHandler #Authentication of API

consumer_key = 'mnCscr8qMPKXZrB6VIlCo5DUr' # your twitter consumer_key
consumer_secret = '2hWiO5AkAa9EMaMDTPQ6bJhwk8xE4p9Ee5gjeCIJ5gUah7Qfz1' # your twitter consumer_secret key
access_token = '858194321362780160-9WvJhq94NrrWuhV2NdgP8Nj3thhrQe2' # your twitter access_token key
access_secret = 'AcILcUcJe6FBNlefhGWFzkpPFvxjjCl6ZGHcbqoMXhRiN' # your twitter access_secret key

auth = OAuthHandler(consumer_key,consumer_secret) # Verifying whether the keys are valuable to authenticate with twitter
auth.set_access_token(access_token,access_secret) # Verifying whether the keys are valuable to fetch tweets from twitter

args = ['Donald Trump'] # tweets are fetched based on this keyword
api = tweepy.API(auth,timeout=10) # authenticating with the twitter account

# Fetching and appending tweets to a single list

list_tweets=[]
query=args[0]

if (len(args)==1):
    for tweet in tweepy.Cursor(api.search,q=query+" -filter:retweets",lang='en',result_type='recent').items(100):
        # Cursor -> function to fetch tweets from twitter
        # result_type -> fetches recent tweets
        # items() -> tweets fetch limit
        list_tweets.append(tweet.text)

# Using the Text_Classification model to analyse the sentiment of the tweets
        
with open('tfidfmodel.pickle','rb') as f:
    tfidfVectorizer = pickle.load(f) # TFIDF Vectorizer obtained from Text_Classification.py

with open('sentiment_classifer.pickle','rb') as f:
    clf = pickle.load(f) # Logistic model classification obtained from Text_Classification.py

# Preprocessing the fetched tweets using regular expressions

total_pos = 0
total_neg = 0

for tweet in list_tweets:
    tweet = re.sub(r"^https://t.co/[a-zA-Z0-9]*\s", " ", tweet) # removes urls at the beginning of the tweets
    tweet = re.sub(r"\s+https://t.co/[a-zA-Z0-9]*\s", " ", tweet) # removes urls in the middle of the tweets
    tweet = re.sub(r"\s+https://t.co/[a-zA-Z0-9]*$", " ", tweet) # removes urls at the end of the tweets
    tweet = tweet.lower() # converts tweets to lower case
    #replaces all short form of the words with its full form
    tweet = re.sub(r"that's","that is",tweet)
    tweet = re.sub(r"there's","there is",tweet)
    tweet = re.sub(r"what's","what is",tweet)
    tweet = re.sub(r"where's","where is",tweet)
    tweet = re.sub(r"it's","it is",tweet)
    tweet = re.sub(r"who's","who is",tweet)
    tweet = re.sub(r"i'm","i am",tweet)
    tweet = re.sub(r"she's","she is",tweet)
    tweet = re.sub(r"he's","he is",tweet)
    tweet = re.sub(r"they're","they are",tweet)
    tweet = re.sub(r"who're","who are",tweet)
    tweet = re.sub(r"ain't","am not",tweet)
    tweet = re.sub(r"wouldn't","would not",tweet)
    tweet = re.sub(r"shouldn't","should not",tweet)
    tweet = re.sub(r"can't","can not",tweet)
    tweet = re.sub(r"couldn't","could not",tweet)
    tweet = re.sub(r"won't","will not",tweet)
    tweet = re.sub(r"\W"," ",tweet) # replaces all puntuations in sentences with space
    tweet = re.sub(r"\d"," ",tweet) # replaces all numbers in sentences with space
    tweet = re.sub(r"\s+[a-z]\s+"," ",tweet) # replaces all single letter characters in middle of tweets with space
    tweet = re.sub(r"\s+[a-z]$"," ",tweet) # replaces all single letter characters at the end of tweets with space
    tweet = re.sub(r"^[a-z]\s+"," ",tweet) # replaces all single letter characters in the beginning of tweets with space
    tweet = re.sub(r"\s+"," ",tweet) # replaces all multiple spaces in the tweets with single space
    sent = clf.predict(tfidfVectorizer.transform([tweet]).toarray())
    #print(tweet,":",sent)
    if(sent[0] == 0):
        total_neg +=1
    else:
        total_pos+=1

# Plotting a bar graph showing the number of positive and negative tweets based on the specified keyword

import matplotlib.pyplot as plt
import numpy as np
objects = ['Positive','Negative']
y_pos = np.arange(len(objects))

plt.bar(y_pos,[total_pos,total_neg],alpha=0.5)
plt.xticks(y_pos,objects)
plt.ylabel('Number')
plt.title('Number of Postive and Negative Tweets')

plt.show()