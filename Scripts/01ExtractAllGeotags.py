# extract all geocoded tweets

import pandas as pd
import os

path = '/Users/darcy/University/UofA/PhD/EAS/PhDWork/Articles/COVID19'

df = pd.read_pickle(os.path.join(path, 'RawData', 'GeotaggedPickle')).drop_duplicates()
df = df.loc[df['geo.coordinates'].notnull()]

df['lat'] = df['geo.coordinates'].apply(lambda x: x.split(',')[0].replace('{', ''))
df['lng'] = df['geo.coordinates'].apply(lambda x: x.split(',')[1].replace('}', ''))

df.to_csv(os.path.join(path, 'Data', 'AllGeotags.csv'))
