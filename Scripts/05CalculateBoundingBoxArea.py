# calculate areas for bounding box tweets
# write out all tweets with a bound box less than 500 kmsq

import pandas as pd
import os
import numpy as np
import json
import geojson
from shapely.geometry import Polygon
import geopandas as gpd
import shapely.wkt
import fiona
import matplotlib.pyplot as plt

path = '/Users/darcy/University/UofA/PhD/EAS/PhDWork/Articles/COVID19'

df = pd.read_pickle(os.path.join(path, 'Data', 'FinalCleanedTweetsTopicScores'))
print("{} tweets loaded".format(len(df))) # df contains 340858 tweets with full text
try:
  idx = df.index.duplicated()
  values, counts = np.unique(idx, return_counts=True)
  foo = dict(zip(values, counts))
  print("Removing {} duplicates".format(foo[True]))
  df = df.loc[~idx]
  print("{} tweets remain".format(len(df)))
except:
  print("No duplicate tweets found")

def get_wkt(row):
  o = row['place.bounding_box.coordinates'].replace('{', '').replace('}', '').split(',')
  s = [float(x) for x in o]
  polygon = Polygon([(s[0], s[1]), (s[2], s[3]), (s[4], s[5]), (s[6], s[7])])
  return polygon.wkt

df['WKTgeom'] = df.apply(get_wkt, axis=1)

# change CRS from WGS84 to StatsCan Lambert conical
geometry = df['WKTgeom'].map(shapely.wkt.loads)
crs = 'epsg:4326'
geoDF = gpd.GeoDataFrame(df, crs=crs, geometry=geometry)
new_crs = 'epsg:3347'
geoDF = geoDF.to_crs(new_crs)
geoDF['area_sqm'] = geoDF.area
geoDF['area_sqkm'] = geoDF.area/1000000
geoDF.loc[geoDF['area_sqkm'] < 1000, 'area_sqkm'].hist(bins=100)
plt.title("Histogram of area of bounding boxes in COVID-19 tweets")
plt.xlabel('Area (square kilometres)')
plt.show()

# write out the Twitter scores and other select columns
columns = [x for x in geoDF.columns if 'similarity' in x]
geoDF.loc[geoDF['area_sqkm'] < 500, ['created_at', 'full_text', 'place.name', 'WKTgeom', 'geometry', 'area_sqkm'] + columns].to_file(os.path.join(path, 'Data', 'TweetsWithTopicScoresUnder500sqkmStatsCanLambert.gpkg'), layer='tweets', driver="GPKG")
## file written out contains 210414 tweets

# Pickle just tweets under 500 sqkm
geoDF.loc[geoDF['area_sqkm'] <= 500].to_pickle(os.path.join(path, 'RawData', 'TweetsWithTopicScoresUnder500sqkm'))
## Canada has 13756 tweets
## United States has 183369 tweets
