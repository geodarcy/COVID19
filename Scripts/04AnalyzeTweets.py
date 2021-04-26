# analyze COVID-19 tweets
# word2vec model was created using collected tweets
# gt200 model is the Stanford Twitter model

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import datetime
from gensim.models import Word2Vec
import gensim.downloader as api

path = '/Users/darcy/University/UofA/PhD/EAS/PhDWork/Articles/COVID19'

## load a pretrained models
our_w2v_model = Word2Vec.load(os.path.join(path, 'Data', 'TwitterWord2VecModel'))
gt200 = api.load('glove-twitter-200')

#load word lists
topic_lists_DF = pd.read_csv(os.path.join(path, 'CodedTopics', 'Model_Word_Lists_20200712_Final.csv'))
topic_lists_DF = topic_lists_DF.iloc[:,1:]
topics = [x.replace(' ', '_') for x in topic_lists_DF.columns]
topic_lists_DF.columns = topics

# load tweets
df = pd.read_pickle(os.path.join(path, 'Data', 'FinalCleanedTweets')) # 363926 tweets
idx = df.index.duplicated()
values, counts = np.unique(idx, return_counts=True)
foo = dict(zip(values, counts))
print("Removing {} duplicates".format(foo[True]))
df = df.loc[~idx]
print("{} tweets remain".format(len(df)))
columns = df.columns[:list(df.columns).index('gt200_words') + 1]
df = df[columns]
df.drop('self_similarity', axis=1, inplace=True) # 340858

# keep only words appearing in the Stanford model
text_list_dict = {}
for topic in topics:
  text_list_dict[topic] = [x for x in topic_lists_DF[topic] if x in gt200.vocab]

# calculate similarity
for topic in topics:
  print("Calculating {}".format(topic))
  df['self_' + topic + '_similarity'] = df['gt200_words'].apply(lambda x: our_w2v_model.wv.n_similarity(x, text_list_dict[topic]) if len(x) else None)
  df['gt200_' + topic + '_similarity'] = df['gt200_words'].apply(lambda x: gt200.n_similarity(x, text_list_dict[topic]) if len(x) else None)

# write out tweets
df.to_pickle(os.path.join(path, 'Data', 'FinalCleanedTweetsTopicScores'))
