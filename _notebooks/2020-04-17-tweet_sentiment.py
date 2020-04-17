#!/usr/bin/env python
# coding: utf-8

# In[160]:


import tweepy as tw #library for accessing Twitter API
import pandas as pd #data analysis API
import json


# In[161]:


#Twitter App Auth
consumer_key = '41zvYrOHdiIKgSq7Xf5tbTyrp'
consumer_secret = '3rOqNnBRjjqkEpGfQcmyD9kh6WgGALmTmiI6IJUgILee0Z0Uad'
access_key = '1224090842106757120-gD25mu7R2pNTCQCozu5o9SCSpE8XHG'
access_secret = 'D63jeYIxtByfzbMvwr2SSXycTSyR4Hdm1xCxcAt3mNfY5'


# In[162]:


# Initialize API
auth = tw.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key, access_secret)
api = tw.API(auth, wait_on_rate_limit=True)


# In[163]:


# Search terms
search_words = ["#coronavirus", "#COVID19", "#CoronavirusOutbreak"]
date_since = "2020-04-01"


# In[164]:


# Collect tweets
tweets = tw.Cursor(api.search,
              q=search_words,
              lang="en",
              since=date_since, tweet_mode='extended',
              include_rts=True).items(5000)

tweets_arr = []


# In[165]:


# Iterate and print tweets
for tweet in tweets:
    tweets_arr.append(tweet.full_text)
print("Done")


# In[166]:


#Creating data frame of tweets
df_tweets = pd.DataFrame(tweets_arr)
df_tweets


# In[167]:


# NLP libraries
import nltk
from nltk.tokenize import sent_tokenize #sentence tokenization:break text into sentences
from nltk.tokenize import word_tokenize #word tokenization:break sentences into words
from nltk.corpus import stopwords #removal of stop words
from nltk.stem import PorterStemmer #lexicon Normalisation/Stemming: retain only root form of the word


# In[168]:


#Using NLTK package to conduct sentiment analysis
nltk.download('vader_lexicon')


# In[169]:


from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()
scores = []
for tweet in tweets_arr:
    score = sid.polarity_scores(tweet)
    scores.append(score)


# In[170]:


#Dataframe for sentiment scores
df_sentiments = pd.DataFrame(scores)
df_sentiments


# In[171]:


dataset = pd.concat([df_tweets, df_sentiments], axis=1, join='inner')
dataset


# In[172]:


# Generate overall_sentiment using pandas 
overall_sentiment = [] 
for value in dataset["compound"]: 
    if value > 0: 
        overall_sentiment.append("Positive") 
    elif value < 0: 
        overall_sentiment.append("Negative") 
    else: 
        overall_sentiment.append("Neutral") 
       
dataset["overall_sentiment"] = overall_sentiment    
print(dataset)


# In[173]:


data = dataset.drop(columns ={"neg","pos","neu","compound"})


# In[174]:


# changing cols with rename() 
data = data.rename(columns = {0: "text"})
print(data)


# In[175]:


#visualizing word clouds for positive and negative tweets

from wordcloud import WordCloud,STOPWORDS
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

data_pos = data[ data['overall_sentiment'] == 'Positive']
data_pos = data_pos['text']
data_neg = data[ data['overall_sentiment'] == 'Negative']
data_neg = data_neg['text']
data_neu = data[ data['overall_sentiment'] == 'Neutral']
data_neu = data_neu['text']

def wordcloud_draw(data, color = 'black'):
    words = ' '.join(data)
    cleaned_word = " ".join([word for word in words.split()
                            if 'http' not in word
                                and not word.startswith('@')
                                and not word.startswith('#')
                                and word != 'RT'
                            ])
    wordcloud = WordCloud(stopwords=STOPWORDS,
                      background_color=color,
                      width=2500,
                      height=2000
                     ).generate(cleaned_word)
    plt.figure(1,figsize=(9, 9))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()
    
print("Positive words")
wordcloud_draw(data_pos,'white')
print("Negative words")
wordcloud_draw(data_neg)
print("Neutral words")
wordcloud_draw(data_neu)


# In[176]:


#import machine learning libraries

import time
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn import model_selection, naive_bayes, metrics, svm
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


# In[177]:


#Splitting the data in train and test split
t1 = time.time()
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['overall_sentiment'], test_size=0.3,random_state = 0)
t2= time.time()

print(round(t2-t1, 2)," secs")


# In[178]:


# Represent the review text as a bag-of-words 
# Text preprocessing, tokenizing and filtering of stopwords are all included in CountVectorizer, which builds a dictionary of features and transforms documents to feature vectors

count_vect = CountVectorizer(lowercase=True,stop_words="english",min_df=10)
count_vect.fit(X_train)

X_train_counts = count_vect.transform(X_train)
X_test_counts = count_vect.transform(X_test)

# Create the tf-idf representation using the bag-of-words matrix

tfidf_transformer = TfidfTransformer(norm=None)
tfidf_transformer.fit(X_train_counts)

X_train_tfid =tfidf_transformer.transform(X_train_counts)
X_test_tfid = tfidf_transformer.transform(X_test_counts)


# In[179]:


X_train_counts.shape


# In[180]:


X_test_counts.shape


# In[181]:


#Applying Naive Bayes Classifier

from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()

#fit data to NB model

nb.fit(X_train_tfid, y_train) # train naive bayes on count

print("Train Accuracy: ", round(nb.score(X_train_tfid,y_train), 3))
print("Test Accuracy: ", round(nb.score(X_test_tfid,y_test), 3))


# In[182]:


#predicting the sentiment of a new tweet

docs_new = ['Balancing working from home and shouldering the bulk of domestic tasks leaves many women stretched to capacity, meaning less quality time with their families and for themselves.']
X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

clf = nb.fit(X_train_tfid, y_train)

predicted = clf.predict(X_new_tfidf)
print(predicted)

