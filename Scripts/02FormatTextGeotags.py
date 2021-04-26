# format text of geotagged tweets

import pandas as pd
import os
import matplotlib.pyplot as plt
import re

path = '/Users/darcy/University/UofA/PhD/EAS/PhDWork/Articles/COVID19'

df = pd.read_pickle(os.path.join(path, 'RawData', 'GeotaggedPickle'))
df = df.loc[(df['place.country'] == 'Canada') | (df['place.country'] == 'United States')]
df = df.loc[df['full_text'].notnull()]
# df contains 46920 tweets

# try to find URLs at the end of the tweet and remove them
re_exp = re.compile('. https://t.co/[\w]+\Z')
df['text_noURL'] = df['full_text'].apply(lambda x: re_exp.sub('', x))
# check how many characters were stipped off
df['text_length'] = df['full_text'].apply(lambda x: len(x))
df['text_noURL_length'] = df['text_noURL'].apply(lambda x: len(x))
df['text_length_diff'] = df['text_length'] - df['text_noURL_length']
df['text_length_diff'].value_counts()
df['text_noURL_length'].hist(bins=20)
df['text_noURL_length'].describe()

df.loc[df['text_noURL'] == "Join the discussion about re-opening schools.  Download your FREE 158 page eBook written by 16 educational experts by visiting:\n\nhttps://t.co/w4sCfqHgfV\n\n#pushboundEDU #COVID_19 #coronavirus #corona #education #schools #SEL #mentalhealth #schoolreform #technolog", 'user.name'].value_counts()
