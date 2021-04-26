## This script is the start of the rewrite for the COVID paper for TR-A
## First step in the analysis is to find related vectors to most common gt200_words
## Using all geotagged tweets in Canada and U.S. between 1 April and 1 July 2020
# format text of select bounding box tweets

import pandas as pd
import os
import matplotlib.pyplot as plt
import re
import datetime
import matplotlib.dates as mdates
from nltk.corpus import stopwords
import string
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.models import Word2Vec
import gensim.downloader as api
from collections import Counter
## load a pretrained model
## for revise and resubmit, not using pretrained model
# gt200 = api.load('glove-twitter-200')

path = '/Users/darcy/University/UofA/PhD/EAS/PhDWork/Articles/COVID19'

df = pd.read_pickle(os.path.join(path, 'Data', 'BoundingBoxesForTextAnalysis')).drop_duplicates()
# df contains 363926 tweets

## create dates
df['dtDate'] = pd.to_datetime(df['created_at'])
## store just the day of the tweet for mapping purposes
df['TweetDay'] = df['dtDate'].dt.date
start_date = datetime.date(2020, 4, 1)
end_date = datetime.date(2020, 7, 1)
total_days = end_date - start_date
df = df.loc[(df['TweetDay'] >= start_date) & (df['TweetDay'] <= end_date)]
total_days = df['TweetDay'].max() - df['TweetDay'].min()

# start to analyze tweets
## add a few extra stopwords
stopwords = list(set(stopwords.words('english')))
newStopwords = ['covid', 'amp', 'rt', 'https', 'covid19', 'coronavirus', 'socialdistancing', '@', 'stayhome']
stopwords.extend(newStopwords)
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()

def process_raw_text(text):
  valid = u"0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ#@/ äöåÄÖÅ"
  url_match = "(https?:\/\/[0-9a-zA-Z\-\_]+\.[\-\_0-9a-zA-Z]+\.?[0-9a-zA-Z\-\_]*\/?.*)"
  name_match = "\@[\_0-9a-zA-Z]+\:?"
  text = re.sub(url_match, u"", text)
  text = re.sub(name_match, u"", text)
  text = re.sub("\&amp\;?", u"", text)
  text = re.sub("[\:\.]{1,}$", u"", text)
  text = re.sub("^RT\:?", u"", text)
  text = u''.join(x for x in text if x in valid)
  text = text.strip()
  return text

def tokenize_tweet(text):
  ret = []
  words = re.split(r'(\s+)', text)
  if len(words) > 0:
    for w in words:
      if w is not None:
        w = w.strip()
        w = w.lower()
        if w.isspace() or w == "\n" or w == "\r":
          w = None
        if len(w) < 1:
          w = None
        if w is not None:
          ret.append(w)
  return ret

def clean_sentences(tokens):
  ret = []
  for token in tokens:
    if len(token) > 0:
      if stopwords is not None:
        for s in stopwords:
          if token == s:
            token = None
      if token is not None:
          if re.search("^[0-9\.\-\s\/]+$", token):
            token = None
      if token is not None:
          ret.append(token)
  return ret

def get_word_frequencies(corpus):
  frequencies = Counter()
  for sentence in corpus:
    for word in sentence:
      frequencies[word] += 1
  freq = frequencies.most_common()
  return freq

# remove words in the tweets that are not found in the glove-twitter-200 model
# def remove_nonvocab_words(tweet):
#   return [x for x in tweet if x in gt200.vocab]

df['processed_tweet'] = df['full_text'].apply(process_raw_text)
df['tokenize_tweet'] = df['processed_tweet'].apply(tokenize_tweet)
df['clean_tweet'] = df['tokenize_tweet'].apply(clean_sentences)
# df['gt200_words'] = df['clean_tweet'].apply(remove_nonvocab_words)
# save dataset
df.to_pickle(os.path.join(path, 'Data', 'FinalCleanedTweets_20210327'))

# prepare a list of list of tweets for Word2Vec
sentences = list(df['clean_tweet'])
frequencies = get_word_frequencies(sentences)
word2vec = Word2Vec(min_count=1, vector_size=50, workers=7, window=5, sg=1, sample=0)
word2vec.build_vocab(sentences)
word2vec.train(sentences, total_examples=len(sentences), epochs=10)
# save Word2Vec model
word2vec.save(os.path.join(path, 'Data', 'TwitterWord2VecModel'))

# create scoring table for Manish and Damian
test_words = ['sidewalk', 'dining', 'closed', 'curb', 'curbside', 'restaurant']
test_words_df = pd.DataFrame()
for word in test_words:
  similar_words = word2vec.wv.most_similar(word)
  foo = pd.DataFrame(similar_words, columns=['similar word', 'similarity score'])
  foo.index = [word] * 10
  test_words_df = test_words_df.append(foo)
test_words_df.to_csv(os.path.join(path, 'Similar_Words20210327.csv'))
