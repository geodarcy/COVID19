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

# load word lists
topic_lists_DF = pd.read_csv(os.path.join(path, 'CodedTopics', 'Model_Curbside_Word_List.csv'))
# rearrange into one list
topic_lists_DF = topic_lists_DF.iloc[:,1:]
topics = topic_lists_DF.columns
curbside_list = list(np.unique([item for topic in topics for item in topic_lists_DF[topic] if item in gt200.vocab]))

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

# calculate similarity
df['self_curbside_similarity'] = df['gt200_words'].apply(lambda x: our_w2v_model.wv.n_similarity(x, curbside_list) if len(x) else None)
df['gt200_curbside_similarity'] = df['gt200_words'].apply(lambda x: gt200.n_similarity(x, curbside_list) if len(x) else None)

# write out tweets
df.to_pickle(os.path.join(path, 'Data', 'FinalCleanedTweetsTopicScores'))
