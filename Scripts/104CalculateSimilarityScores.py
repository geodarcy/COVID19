import pandas as pd
import os
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
import numpy as np

path = '/Users/darcy/University/UofA/PhD/EAS/PhDWork/Articles/COVID19'

word2vec = Word2Vec.load(os.path.join(path, 'Data', 'TwitterWord2VecModel'))
df = pd.read_pickle(os.path.join(path, 'Data', 'FinalCleanedTweets_20210327'))
idx = df.index.duplicated()
values, counts = np.unique(idx, return_counts=True)
foo = dict(zip(values, counts))
print("Removing {} duplicates".format(foo[True]))
df = df.loc[~idx]
print("{} tweets remain".format(len(df)))

# concept words as determined with Manish and Damian
# test individual words with most_similar
# word2vec.wv.most_similar('permit')
# permit doesn't fit well
entertainment_words = ['dining', 'restaurant', 'business', 'permit', 'customer', 'retail']
# close changed to closed
transportation_words = ['walk', 'bike', 'closed', 'vehicle', 'traffic']
# expand doesn't fit well
# curb changed to curb
# convert doesn't fit well
spaces_words = ['space', 'outdoor', 'lane', 'street', 'parking', 'sidewalk', 'expand', 'curbside', 'convert']

# calculate similarity scores
df['entertainment_similarity'] = df['clean_tweet'].apply(lambda x: word2vec.wv.n_similarity(x, entertainment_words) if len(x) else None)
df['transportation_similarity'] = df['clean_tweet'].apply(lambda x: word2vec.wv.n_similarity(x, transportation_words) if len(x) else None)
df['spaces_similarity'] = df['clean_tweet'].apply(lambda x: word2vec.wv.n_similarity(x, spaces_words) if len(x) else None)

df = df.sort_values('entertainment_similarity', ascending=False)
df['entertainment_rank'] = np.arange(len(df)) + 1
df = df.sort_values('transportation_similarity', ascending=False)
df['transportation_rank'] = np.arange(len(df)) + 1
df = df.sort_values('spaces_similarity', ascending=False)
df['spaces_rank'] = np.arange(len(df)) + 1

output_columns = ['created_at', 'place.name', 'full_text', 'entertainment_similarity', 'entertainment_rank', 'transportation_similarity', 'transportation_rank', 'spaces_similarity', 'spaces_rank']

df.loc[(df['entertainment_rank'] < 501) | (df['transportation_rank'] < 501) | (df['spaces_rank'] < 501), output_columns].to_csv(os.path.join(path, 'Similarity_Scored_Tweets_Revision.csv'))

# calculate similarity scores again
cycling_words = ['cycling', 'bike', 'curbside', 'sidewalk']
driving_words = ['parking', 'car', 'space']
walking_words = ['walking', 'sidewalk']
business_words = ['restaurant', 'patio', 'outdoor', 'seating', 'dining']

df['cycling_similarity'] = df['clean_tweet'].apply(lambda x: word2vec.wv.n_similarity(x, cycling_words) if len(x) else None)
df['driving_similarity'] = df['clean_tweet'].apply(lambda x: word2vec.wv.n_similarity(x, driving_words) if len(x) else None)
df['walking_similarity'] = df['clean_tweet'].apply(lambda x: word2vec.wv.n_similarity(x, walking_words) if len(x) else None)
df['business_similarity'] = df['clean_tweet'].apply(lambda x: word2vec.wv.n_similarity(x, business_words) if len(x) else None)

df = df.sort_values('cycling_similarity', ascending=False)
df['cycling_rank'] = np.arange(len(df)) + 1
df = df.sort_values('driving_similarity', ascending=False)
df['driving_rank'] = np.arange(len(df)) + 1
df = df.sort_values('walking_similarity', ascending=False)
df['walking_rank'] = np.arange(len(df)) + 1
df = df.sort_values('business_similarity', ascending=False)
df['business_rank'] = np.arange(len(df)) + 1

output_columns = ['created_at', 'place.name', 'full_text', 'cycling_similarity', 'cycling_rank', 'driving_similarity', 'driving_rank', 'walking_similarity', 'walking_rank', 'business_similarity', 'business_rank']

df.loc[(df['cycling_rank'] < 501) | (df['driving_rank'] < 501) | (df['walking_rank'] < 501) | (df['business_rank'] < 501), output_columns].to_csv(os.path.join(path, 'Similarity_Scored_Tweets_Revision_20210414.csv'))

# try just one list for curbside
curbside_columns = ['created_at', 'place.name', 'full_text', 'curbside_similarity', 'curbside_rank']
curbside_words = ['curbside', 'bike', 'restaurant', 'patio', 'parking', 'lane']
df['curbside_similarity'] = df['clean_tweet'].apply(lambda x: word2vec.wv.n_similarity(x, curbside_words) if len(x) else None)
df = df.sort_values('curbside_similarity', ascending=False)
df['curbside_rank'] = np.arange(len(df)) + 1
df.loc[df['curbside_rank'] < 501, curbside_columns].to_csv(os.path.join(path, 'Similarity_Scored_Curbside_Tweets_Revision_20210414.csv'))
