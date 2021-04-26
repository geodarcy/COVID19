## read in and summarize topic words that will be used for word vector lists

import pandas as pd
from pandas import DataFrame
import os
import numpy as np
from gensim.models import Word2Vec
import gensim.downloader as api
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm
gt200 = api.load('glove-twitter-200')

path = '/Users/darcy/University/UofA/PhD/EAS/PhDWork/Articles/COVID19/CodedTopics'

damianDF = pd.read_csv(os.path.join(path, 'Score_Top500_Words_20200627_DC_edit.csv'), na_values='?').set_index('word')
darcyDF = pd.read_csv(os.path.join(path, 'Score_Top500_Words_20200626_DR.csv')).set_index('word')
manishDF = pd.read_csv(os.path.join(path, 'Score_Top500_Words_20200627_MS_edit.csv')).set_index('word')

columns = damianDF.columns[3:-1]
DFDict = {}
for column in columns:
  DFDict[column] = DataFrame({'Damian': damianDF[column], 'Darcy':darcyDF[column], 'Manish':manishDF[column]})
  DFDict[column]['Total'] = DFDict[column][['Damian', 'Darcy', 'Manish']].sum(axis=1)
  DFDict[column].sort_values('Total', ascending=False).to_csv(os.path.join(path, 'Summed_' + column + '.csv'))

public_space_list = list(DFDict['public_space'].loc[DFDict['public_space']['Total'] >= 2].index)
mobility_list = list(DFDict['mobility'].loc[DFDict['mobility']['Total'] >= 2].index)
stay_home_list = list(DFDict['stay_home'].loc[DFDict['stay_home']['Total'] == 3].index)
personal_effect_list = list(DFDict['personal_effect'].loc[DFDict['personal_effect']['Total'] >= 2].index)
essential_services_list = list(DFDict['essential_services'].loc[DFDict['essential_services']['Total'] >= 2].index)

# clean public space list
public_space_duplicates = ['closed', 'communities', 'distancing', 'masks', 'safety']
public_space_list = [x for x in public_space_list if x in gt200.vocab]
public_space_list = [x for x in public_space_list if x not in public_space_duplicates]
gt200.doesnt_match(public_space_list)

# clean mobility space list
mobility_duplicates = ['going', 'workers']
mobility_list = [x for x in mobility_list if x in gt200.vocab]
mobility_list = [x for x in mobility_list if x not in mobility_duplicates]
gt200.doesnt_match(mobility_list)

# clean stay home space list
stay_home_duplicates = ['distancing', 'home.', 'house', 'staying']
stay_home_list = [x for x in stay_home_list if x in gt200.vocab]
stay_home_list = [x for x in stay_home_list if x not in stay_home_duplicates]
gt200.doesnt_match(stay_home_list)

# clean stay personal effect list
personal_effect_duplicates = ['death', 'deaths', 'die', 'died', 'dying', 'families', 'feeling', 'friends', 'helping', 'loved']
personal_effect_list = [x for x in personal_effect_list if x in gt200.vocab]
personal_effect_list = [x for x in personal_effect_list if x not in personal_effect_duplicates]
gt200.doesnt_match(personal_effect_list)

# clean stay essential servies list
essential_services_duplicates = ['businesses', 'dr.', 'hospitals', 'reopening', 'services', 'workers', 'working']
essential_services_list = [x for x in essential_services_list if x in gt200.vocab]
essential_services_list = [x for x in essential_services_list if x not in essential_services_duplicates]
gt200.doesnt_match(essential_services_list)

## additional words
public_space_text_additions = ['street', 'greenspace', 'patio', 'square', 'plaza', 'river', 'fountain', 'alley']
stay_home_text_additions = ['isolate', 'renter', 'schooling', 'telecommuting']
mobility_text_additions = ['bike', 'walk', 'transit', 'pedestrian', 'car', 'lane', 'bus', 'train', 'metro', 'uber', 'lyft', 'scooter', 'taxi']
personal_effect_additions = []
essential_services_additions = []

public_space_list_export = public_space_list + public_space_text_additions
public_space_list_export.sort()
mobility_list_export = mobility_list + mobility_text_additions
mobility_list_export.sort()
stay_home_list_export = stay_home_list + stay_home_text_additions
stay_home_list_export.sort()
personal_effect_list_export = personal_effect_duplicates + personal_effect_additions
personal_effect_list_export.sort()
essential_services_list_export = essential_services_list + essential_services_additions
essential_services_list_export.sort()

outDF = DataFrame.from_dict({'public space': public_space_list_export, 'mobility': mobility_list_export, 'stay home': stay_home_list_export, 'personal effect': personal_effect_list_export, 'essential services': essential_services_list_export}, orient='index').T
outDF.to_csv(os.path.join(path, 'Model_Word_Lists.csv'))
