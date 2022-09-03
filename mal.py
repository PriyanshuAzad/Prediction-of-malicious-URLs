data_dir = "E:\Project_2\Machine-Learning-for-Security-Analysts-master\Malicious URLs.csv"

# common imports
from cProfile import label
from cgi import test

from itertools import count
from tkinter import font
from turtle import color
from django import urls
from django.forms import model_to_dict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import re
import pickle

import streamlit as st
# matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


#import scikit-learn models
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB


#import scikit-learn metric function
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns






url_df = pd.read_csv(data_dir)

test_url = url_df['URLs'][14]



test_percentage = .2

train_df, test_df = train_test_split(url_df, test_size=test_percentage, random_state=42)

labels = train_df['Class']
test_labels = test_df['Class']


def tokenizer(url):
  
  
  tokens = re.split('[/-]', url)
  
  for i in tokens:
  
    if i.find(".") >= 0:
      dot_split = i.split('.')
      
      # Remove .com and www. since they're too common
      if "com" in dot_split:
        dot_split.remove("com")
      if "www" in dot_split:
        dot_split.remove("www")
      
      tokens += dot_split
      
  return tokens



cVec = CountVectorizer(tokenizer=tokenizer)
count_X = cVec.fit_transform(train_df['URLs'])

tVec = TfidfVectorizer(tokenizer=tokenizer)
#vectorizer = TfidfVectorizer(tokenizer=makeTokens)
tfidf_X = tVec.fit_transform(train_df['URLs'])


test_count_X = cVec.transform(test_df['URLs'])

test_tfidf_X = tVec.transform(test_df['URLs'])
#need




mnb_count = MultinomialNB(alpha = .1)
model = mnb_count.fit(count_X, labels)


# x_predict = ["https://cs.stanford.edu/people/nick/py/python-print.html"]

# x_predict= tVec.transform(x_predict)
# New_predict = mnb_count.predict(x_predict) 

# print(New_predict) 






import pickle
pickle_out = open("classifier.pkl", "wb")
pickle.dump(model, pickle_out)
pickle_out.close()













 