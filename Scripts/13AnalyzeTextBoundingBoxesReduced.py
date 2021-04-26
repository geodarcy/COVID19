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
gt200 = api.load('glove-twitter-200')

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
plt.close()
df['TweetDay'].hist(bins=total_days.days)
plt.xlabel('Date')
plt.ylabel('Number of geolocated tweets')
locs, labels = plt.xticks()
plt.xticks(ticks=locs[::2], labels=labels[::2])
plt.savefig(os.path.join(path, 'Images', 'TweetsPerDayReduced.png'), bbox_inches='tight', dpi=300)
plt.close()
# timeBins = [datetime.datetime.strptime(x, '%Y-%m-%d') for x in ['2017-08-26', '2017-09-07', '2017-09-11', '2017-09-13', '2017-10-04']]
# labels=['t-1', 't0', 't1', 't2']
# df['TimeBins'] = pd.cut(df['dtDate'], timeBins, labels=labels)
# for label in labels:
#   print('{} has {} tweets'.format(label, len(df[df['TimeBins'] == label])))
#   df[df['TimeBins'] == label]['dtDate'].groupby(df.dtDate.dt.date).count().plot(kind='bar', color='blue')
#   plt.show()

# start to analyze tweets
## add a few extra stopwords
stopwords = list(set(stopwords.words('english')))
newStopwords = ['covid', 'amp', 'rt', 'https', 'covid19', 'coronavirus', 'socialdistancing', '@', 'stayhome']
stopwords.extend(newStopwords)
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()

def process_raw_text(text):
  valid = u"0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ#@.:/ äöåÄÖÅ"
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
def remove_nonvocab_words(tweet):
  return [x for x in tweet if x in gt200.vocab]

df['processed_tweet'] = df['full_text'].apply(process_raw_text)
df['tokenize_tweet'] = df['processed_tweet'].apply(tokenize_tweet)
df['clean_tweet'] = df['tokenize_tweet'].apply(clean_sentences)
df['gt200_words'] = df['clean_tweet'].apply(remove_nonvocab_words)
# save dataset
df.to_pickle(os.path.join(path, 'Data', 'FinalCleanedTweets'))

# prepare a list of list of tweets for Word2Vec
sentences = list(df['clean_tweet'])
frequencies = get_word_frequencies(sentences)
word2vec = Word2Vec(min_count=1, size=50, workers=7, window=5, sg=1, sample=0)
word2vec.build_vocab(sentences)
word2vec.train(sentences, total_examples=len(sentences), epochs=10)
# save Word2Vec model
word2vec.save(os.path.join(path, 'Data', 'TwitterWord2VecModel'))

# check similarity to a test list of words
public_space_text = ['access', 'beach', 'close', 'community', 'contact', 'distance', 'local', 'mask', 'open', 'outside', 'park', 'place', 'play', 'public', 'run', 'safe', 'social']
public_space_text_additions = ['street', 'greenspace', 'patio', 'square', 'plaza', 'river', 'fountain', 'alley']
stay_home_text = ['distance', 'home', 'lockdown', 'quarantine', 'stay']
stay_home_text_additions = ['isolate', 'renter', 'schooling', 'telecommuting']
mobility_text = ['contact', 'free', 'go', 'outside', 'run', 'social', 'visit', 'walk', 'working']
mobility_text_additions = ['bike', 'walk', 'transit', 'pedestrian', 'car', 'lane', 'bus', 'train', 'metro', 'uber', 'lyft', 'scooter', 'taxi']
personal_effect = ['children', 'dead', 'family', 'feel', 'friend', 'happy', 'hard', 'help', 'hope', 'impact', 'kids', 'love', 'sick', 'symptoms']
personal_effect_additions = []
essential_services = ['access', 'business', 'doctors', 'employees', 'essential', 'food', 'frontline', 'grocery', 'healthcare', 'hospital', 'job', 'medical', 'nurses', 'patients', 'reopen', 'resources', 'response', 'service', 'staff', 'store', 'tests', 'work']
essential_services_additions = []

text_list_dict = {'public_space': public_space_text, 'stay_home': stay_home_text, 'mobility': mobility_text}

for key in text_list_dict:
  df['self_' + key + '_similarity'] = df['gt200_words'].apply(lambda x: word2vec.wv.n_similarity(x, text_list_dict[key]) if len(x) else None)
  df['gt200_' + key + '_similarity'] = df['gt200_words'].apply(lambda x: gt200.n_similarity(x, text_list_dict[key]) if len(x) else None)
# df['gt200_similarity'].hist(bins=50)
# print(df[['gt200_similarity', 'self_similarity']].corr())
plt.scatter(df['gt200_similarity'], df['self_similarity'])
plt.xlabel('GlOve Twitter 200')
plt.ylabel('Model from collected tweets')

# check results
list(df.sort_values('self_public_space_similarity', ascending=False)['full_text'])[:10]

# create scoring table for Manish and Damian
outDF = pd.DataFrame(frequencies[:500], columns=['word', 'frequency'])
outDF['in_Stanford_model'] = outDF['word'].apply(lambda x: 'True' if (x in gt200.vocab) else 'False')
for i in ['mobility', 'stay_home', 'essential_services', 'public_space', 'personal_effect', 'health']:
  outDF[i] = None
# outDF.to_csv(os.path.join(path, 'Score_Top500_Words_20200626.csv'))
