# cluster tweet scores

import pandas as pd
from pandas import DataFrame
import os
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
import fiona
import datetime
import matplotlib.dates as mdates

pd.set_option('mode.chained_assignment', None)
months = mdates.MonthLocator()
month_fmt = mdates.DateFormatter('%b')

path = '/Users/darcy/University/UofA/PhD/EAS/PhDWork/Articles/COVID19'

df = pd.read_pickle(os.path.join(path, 'RawData', 'TweetsWithTopicScoresUnder500sqkm'))
print("{} tweets read in".format(len(df))) # 197125 tweets
try:
  idx = df.index.duplicated()
  values, counts = np.unique(idx, return_counts=True)
  foo = dict(zip(values, counts))
  print("Removing {} duplicates".format(foo[True]))
  df = df.loc[~idx]
except:
  print("No duplicates found")

# read in urban areas for Canada and the U.S.
# for Canada, Census Metropolitan Areas were used
# for U.S., Urbanized Areas were used
# for U.S., see: https://www.census.gov/cgi-bin/geo/shapefiles/index.php?year=2019&layergroup=Urban+Areas
cmas = pd.read_csv(os.path.join(path, 'Data', 'TweetsWithCMA.csv')).set_index('id')
idx = cmas.index.duplicated()
values, counts = np.unique(idx, return_counts=True)
foo = dict(zip(values, counts))
print("Removing {} duplicates".format(foo[True]))
cmas = cmas.loc[~idx]
us_ua = pd.read_csv(os.path.join(path, 'Data', 'TweetsWithUS_UA.csv')).set_index('id')
idx = us_ua.index.duplicated()
values, counts = np.unique(idx, return_counts=True)
foo = dict(zip(values, counts))
print("Removing {} duplicates".format(foo[True]))
us_ua = us_ua.loc[~idx]

# attach the name of the urban area to the tweets
df['UrbanArea'] = cmas['cmaname']
df.loc[df['UrbanArea'].isnull(), 'UrbanArea'] = us_ua['name10'] # 181851 tweets in urban areas

# need to find
urban_areas = df['UrbanArea'].dropna().unique()
cities = ['Portland', 'San Francisco', 'Washington', 'New York', 'Vancouver', 'Chicago', 'Pittsburg', 'Toronto']
for city in cities:
  print("\n".join(s for s in urban_areas if city.lower() in s.lower()))
# copy names from terminal
final_cities = ['Portland, OR--WA', 'San Francisco--Oakland, CA', 'Washington, DC--VA--MD', 'New York--Newark, NY--NJ--CT', 'Vancouver', 'Chicago, IL--IN', 'Pittsburgh, PA', 'Toronto']
for city in final_cities:
  print("{} has {} tweets".format(city, len(df.loc[df['UrbanArea'] == city])))
# check the number of tweets in a city at various bounding box limits
sizes = range(50,550,50)
sizesDF = DataFrame(index=final_cities, columns=sizes)
for size in sizes:
  sizesDF[size] = df.loc[df['area_sqkm'] <= size, 'UrbanArea'].value_counts()
sizesDF.to_csv(os.path.join(path, 'Data', 'CityBoundingBoxSizes.csv'))

# calculate the mean similarity scores for each city
columns = [x for x in df.columns if 'similarity' in x]
similarityDF = DataFrame(index=['All Tweets'] + final_cities, columns = columns)
similarityDF.loc['All Tweets', columns] = df[columns].mean()
for city in final_cities:
  similarityDF.loc[city, columns] = df.loc[df['UrbanArea'] == city, columns].mean()
similarityDF.to_csv(os.path.join(path, 'Data', 'CityTopicSimilarityScores.csv'))

# write out the top 50 tweets for each city
# for city in final_cities:
#   public_spaceDF = df.loc[(df['UrbanArea'] == city) & (df['area_sqkm'] <= 250)].nlargest(50, 'self_public_space_similarity')
#   mobilityDF = df.loc[(df['UrbanArea'] == city) & (df['area_sqkm'] <= 250)].nlargest(50, 'self_mobility_similarity')
#   curbsideDF = df.loc[(df['UrbanArea'] == city) & (df['area_sqkm'] <= 250)].nlargest(50, 'self_curbside_similarity')
#   outDF = pd.concat([public_spaceDF, mobilityDF, curbsideDF])
#   try:
#     idx = outDF.index.duplicated()
#     values, counts = np.unique(idx, return_counts=True)
#     foo = dict(zip(values, counts))
#     print("Removing {} duplicates in {}".format(foo[True], city))
#     outDF = outDF.loc[~idx]
#   except:
#     print("Writing out all 150 tweets")
#   outDF[['UrbanArea', 'self_public_space_similarity', 'self_mobility_similarity', 'self_curbside_similarity', 'full_text']].to_csv(os.path.join(path, 'CodedTopics', 'CityTweets', city + 'TopTweets.csv'))

# add just tweets in select cities to GeoPackage
# Bounding box limit is 500 sqkm
geometry = df['geometry']
crs = 'epsg:3347' # this is already defined by geometry but reinforcing it here
geoDF = gpd.GeoDataFrame(df, crs=crs, geometry=geometry)
# geoDF.loc[(geoDF['area_sqkm'] < 500) & (geoDF['UrbanArea'].isin(final_cities)), ['created_at', 'full_text', 'place.name', 'WKTgeom', 'geometry', 'area_sqkm', 'UrbanArea'] + columns].to_file(os.path.join(path, 'Data', 'TweetsWithTopicScoresUnder500sqkmStatsCanLambert.gpkg'), layer='final_cities', driver="GPKG")

# calculate a mean score for each check-in location
grouped = geoDF.loc[(geoDF['area_sqkm'] < 500) & (geoDF['UrbanArea'].isin(final_cities))].groupby('place.name')
geo_out_DF = DataFrame(grouped['geometry'].first())
geo_out_DF['UrbanArea'] = grouped['UrbanArea'].first()
geo_out_DF['CheckInCount'] = grouped['UrbanArea'].count()
geo_out_DF[columns] = grouped[columns].mean()
geometry = geo_out_DF['geometry']
crs = 'epsg:3347' # this is already defined by geometry but reinforcing it here
geo_out_DF_out = gpd.GeoDataFrame(geo_out_DF, crs=crs, geometry=geometry)
# geo_out_DF_out.to_file(os.path.join(path, 'Data', 'TweetsWithTopicScoresUnder500sqkmStatsCanLambert.gpkg'), layer='mean_similarities', driver="GPKG")

# calculate Moran's I
from esda.moran import Moran
from esda.moran import Moran_Local
import libpysal as lps
city_dict = dict(zip(cities, final_cities))
analysis_columns = ['self_public_space_similarity', 'self_mobility_similarity', 'self_curbside_similarity']
analysis_dict = dict(zip(['public_space', 'mobility', 'curbside'], analysis_columns))
for city in city_dict.keys():
  print("Running Moran's I for {}".format(city))
  city_DF = geoDF.loc[(geoDF['UrbanArea'] == city_dict[city]) & (city_DF[analysis_dict[analysis]].notnull())]
  city_DF['geometry'] = city_DF['geometry'].centroid
  city_DF['X'] = city_DF['geometry'].centroid.x
  city_DF['Y'] = city_DF['geometry'].centroid.y
  xy_array = city_DF[['X', 'Y']].to_numpy()
  min_dist = lps.weights.min_threshold_distance(xy_array)
  w = lps.weights.DistanceBand.from_dataframe(city_DF, threshold=min_dist, alpha=-2)
  for analysis in analysis_dict.keys():
    print("{}".format(analysis))
    mi = Moran(city_DF[analysis_dict[analysis]], w)
    if mi.p_sim < 0.05:
      print("{} in {} is significant".format(analysis, city))
    else:
      print("{} in {} is NOT significant".format(analysis, city))
    lisa = Moran_Local(city_DF[analysis_dict[analysis]].values, w)
    city_DF['quadrant'] = lisa.q
    city_DF['significant'] = lisa.p_sim < 0.05
    print(city_DF['quadrant'].value_counts())
    print(city_DF['significant'].value_counts())
    layer = city + '_' + analysis + '_lisa'
    city_DF.to_file(os.path.join(path, 'Data', 'TweetsWithTopicScoresUnder500sqkmStatsCanLambert.gpkg'), layer=layer, driver="GPKG")

# write out tweets in selected cities to do Moran's I at the tweet level
out_fields = ['created_at', 'full_text', 'self_public_space_similarity', 'self_mobility_similarity', 'self_curbside_similarity', 'area_sqkm', 'UrbanArea', 'geometry']
for city in cities:
  tempDF = geoDF.loc[geoDF['UrbanArea'] == city_dict[city], out_fields]
  tempDF['geometry'] = tempDF['geometry'].centroid
  tempDF.columns = ['created_at', 'full_text', 'publicsp', 'mobility', 'curbside', 'area_sqkm', 'UrbanArea', 'geometry']
  tempDF.to_file(os.path.join(path, 'Data', 'TweetsForMI', city + 'ForMI.shp'))

# calculate scatter plot of mean curbside value vs day
for city in cities:
  grouped = df.loc[df['UrbanArea'] == city_dict[city]].groupby('TweetDay')
  x = [datetime.date.toordinal(x) for x in grouped['TweetDay'].first()]
  y = grouped['self_curbside_similarity'].mean()
  s = grouped['self_curbside_similarity'].count() / 2
  z = np.polyfit(x, y, 1)
  p = np.poly1d(z)
  plt.scatter(grouped['TweetDay'].first(), y, s=s, label='dot size indicates tweets/d')
  plt.plot(x, p(x), 'r--', label='linear trend')
  plt.plot(x, y.rolling(window=7).mean(), label='7 day moving avg', color='orange')
  plt.title(str.title(city))
  plt.ylabel("Mean Daily Curbside Similarity Score")
  plt.xlim((737490, 737620))
  plt.ylim((0.55, 0.85))
  plt.legend(loc='upper left')
  plt.savefig(os.path.join(path, 'Images', 'CurbsideVsTime', city + 'CurbsideTrend.png'), pad_inches='tight')
  plt.close()

# try to find tweets where the curbside similarity is a maximum
city = 'Toronto'
grouped = df.loc[df['UrbanArea'] == city_dict[city]].groupby('TweetDay')
max_day = foo.loc[foo == foo.max()].index[0]
list(df.loc[(df['TweetDay'] == max_day) & (df['UrbanArea'] == city_dict[city])].sort_values('self_curbside_similarity', ascending=False)['full_text'])
df.loc[(df['TweetDay'] == max_day) & (df['UrbanArea'] == city_dict[city]), 'self_curbside_similarity'].describe()
list(df.loc[df['UrbanArea'] == city_dict[city]].sort_values('self_curbside_similarity', ascending=False)['full_text'])[:10]

# calculate scatter plot of mean curbside value vs day but with a moving average
for city in cities:
  grouped = df.loc[df['UrbanArea'] == city_dict[city]].groupby('TweetDay')
  x = [datetime.date.toordinal(x) for x in grouped['TweetDay'].first()]
  y = grouped['self_curbside_similarity'].mean()
  plt.scatter(grouped['TweetDay'].first(), y, s=0.5, marker='.')
  plt.plot(x, y.rolling(window=7).mean(), label='7 day moving avg', color='orange')
  plt.title(str.title(city))
  plt.ylabel("Mean Daily Curbside Similarity Score")
  plt.ylim((0.55, 0.85))
  plt.legend(loc='upper left')
  plt.savefig(os.path.join(path, 'Images', 'CurbsideVsTimeMA', city + 'CurbsideTrend.png'))
  plt.close()

# redo scatterplot for four cities as one figure
plot_cities = [['Vancouver', 'Toronto'], ['San Francisco--Oakland, CA', 'New York--Newark, NY--NJ--CT']]
plot_cities_short = [['Vancouver', 'Toronto'], ['San Francisco', 'New York']]
fig, ax = plt.subplots(ncols=2, nrows=2, sharex=True, sharey=True, figsize=(7,7))
for i in range(2):
  for j in range(2):
    grouped = df.loc[df['UrbanArea'] == plot_cities[i][j]].groupby('TweetDay')
    x = [datetime.date.toordinal(x) for x in grouped['TweetDay'].first()]
    y = grouped['self_curbside_similarity'].mean()
    s = grouped['self_curbside_similarity'].count() / 2
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    ax[i, j].scatter(grouped['TweetDay'].first(), y, s=s, label='dot size indicates tweets/d')
    ax[i, j].plot(x, p(x), 'r--', label='linear trend')
    ax[i, j].plot(x, y.rolling(window=7).mean(), label='7 day moving avg', color='orange')
    ax[i, j].set_title(str.title(plot_cities_short[i][j]))
ax[i, j].set_xlim((737490, 737620))
ax[i, j].xaxis.set_major_locator(months)
ax[i, j].xaxis.set_major_formatter(month_fmt)
ax[1, 0].legend(loc='lower left', ncol=3, fontsize='small')
ax[1, 0].set_zorder(1)
fig.text(0, 0.5, "Mean Daily Curbside Similarity Score", va='center', rotation='vertical')
fig.savefig(os.path.join(path, 'Images', 'TweetsVsTime.png'), dpi=300, bbox_inches='tight', pad_inches=0.05)

# write out tweet centroids for selected cities
out_fields = ['created_at', 'Month', 'full_text', 'self_public_space_similarity', 'self_mobility_similarity', 'self_curbside_similarity', 'area_sqkm', 'UrbanArea', 'geometry']
tempDF = geoDF.loc[geoDF['UrbanArea'].isin(final_cities)]
tempDF['Month'] = tempDF['TweetDay'].apply(lambda x: x.month)
tempDF = tempDF[out_fields]
tempDF['geometry'] = tempDF['geometry'].centroid
tempDF.columns = ['created_at', 'Month', 'full_text', 'publicspace', 'mobility', 'curbside', 'area_sqkm', 'UrbanArea', 'geometry']
tempDF.to_file(os.path.join(path, 'Data', 'TweetsWithTopicScoresUnder500sqkmStatsCanLambert.gpkg'), layer='tweet_centroids_months', driver="GPKG")
tempDF.to_file(os.path.join(path, 'Data', 'TweetsForMI', 'CentroidsForTemporalMI.shp'))

# export top tweets
for city in cities:
  df.loc[df['UrbanArea'] == city_dict[city], ['TweetDay', 'UrbanArea', 'place.name', 'self_curbside_similarity', 'full_text']].sort_values('self_curbside_similarity', ascending=False).head(250).to_csv(os.path.join(path, 'Paper', 'SampleTweets', city + '_curbside_tweets.csv'))
for city in cities:
  df.loc[df['UrbanArea'] == city_dict[city], ['TweetDay', 'UrbanArea', 'place.name', 'self_public_space_similarity', 'full_text']].sort_values('self_public_space_similarity', ascending=False).head(250).to_csv(os.path.join(path, 'Paper', 'SampleTweets', city + '_public_space_tweets.csv'))
for city in cities:
  df.loc[df['UrbanArea'] == city_dict[city], ['TweetDay', 'UrbanArea', 'place.name', 'self_mobility_similarity', 'full_text']].sort_values('self_mobility_similarity', ascending=False).head(250).to_csv(os.path.join(path, 'Paper', 'SampleTweets', city + '_mobility_tweets.csv'))
# one output for all cities
df[['TweetDay', 'UrbanArea', 'place.name', 'self_mobility_similarity', 'full_text']].sort_values('self_mobility_similarity', ascending=False).head(250).to_csv(os.path.join(path, 'Paper', 'SampleTweets', 'all_mobility_tweets.csv'))

# plot similarity score distribution (Figure 3)
fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True, sharey=True, figsize=(6,6))
ax1.hist(df['self_curbside_similarity'], bins=50)
ax2.hist(df['self_mobility_similarity'], bins=50)
ax3.hist(df['self_public_space_similarity'], bins=50)
ymin, ymax = ax1.get_ylim()
ax1.vlines(df['self_curbside_similarity'].mean(), ymin, ymax, label='mean')
ax2.vlines(df['self_mobility_similarity'].mean(), ymin, ymax)
ax3.vlines(df['self_public_space_similarity'].mean(), ymin, ymax)
ax1.text(0.1, 17500, 'curbside')
ax2.text(0.1, 17500, 'mobility')
ax3.text(0.1, 17500, 'public space')
fig.text(0, 0.5, 'Number of tweets', va='center', rotation='vertical')
fig.text(0.5, 0, 'Similarity Score', ha='center')
ax1.legend()
fig.savefig(os.path.join(path, 'Images', 'TopicHistograms.png'), dpi=300, bbox_inches='tight', pad_inches=0.05)
