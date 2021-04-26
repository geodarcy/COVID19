# extract all bounding box tweets

import pandas as pd
import os
import json
import geojson
from shapely.geometry import Polygon
import geopandas as gpd
import shapely.wkt
import fiona
import matplotlib.pyplot as plt

path = '/Users/darcy/University/UofA/PhD/EAS/PhDWork/Articles/COVID19'

df = pd.read_pickle(os.path.join(path, 'RawData', 'BoundingBoxPickle')).drop_duplicates()
# df contains 372057 tweets with full text

def get_wkt(row):
  o = row['place.bounding_box.coordinates'].replace('{', '').replace('}', '').split(',')
  s = [float(x) for x in o]
  polygon = Polygon([(s[0], s[1]), (s[2], s[3]), (s[4], s[5]), (s[6], s[7])])
  return polygon.wkt

df['WKTgeom'] = df.apply(get_wkt, axis=1)
df.to_csv(os.path.join(path, 'Data', 'AllBoundingBoxes.csv'))

# change CRS from WGS84 to StatsCan Lambert conical
geometry = df['WKTgeom'].map(shapely.wkt.loads)
crs = 'epsg:4326'
geoDF = gpd.GeoDataFrame(df, crs=crs, geometry=geometry)
new_crs = 'epsg:3347'
geoDF = geoDF.to_crs(new_crs)
geoDF['area_sqm'] = geoDF.area
geoDF['area_sqkm'] = geoDF.area/1000000
geoDF.loc[geoDF['area_sqkm'] < 50, 'area_sqkm'].hist(bins=50)
plt.title("Histogram of area of bounding boxes in COVID-19 tweets")
plt.xlabel('Area (square kilometres)')
plt.show()

# geoDF.loc[geoDF.area < 5E7].to_csv(os.path.join(path, 'Data', 'BoundingBoxesUnder50sqkmStatsCanLambert.csv'))
# geoDF.loc[geoDF.area < 5E7].to_file(os.path.join(path, 'Data', 'BoundingBoxesUnder50sqkmStatsCanLambert.shp'))
geoDF.loc[(geoDF['area_sqkm'] < 50) & geoDF['full_text'].notnull()].to_file(os.path.join(path, 'Data', 'BoundingBoxesUnder50sqkmStatsCanLambert.gpkg'), layer='tweets', driver="GPKG")
## file written out contains 56541 tweets

# Pickle just Canada and U.S. tweets under 50 sqkm
geoDF.loc[(geoDF['area_sqkm'] < 50) & geoDF['full_text'].notnull() & ((geoDF['place.country'] == 'Canada') | (geoDF['place.country'] == 'United States'))].to_pickle(os.path.join(path, 'RawData', 'BoundingBoxPickleCanUS50sqkm'))
## Canada has 3220 tweets
## United States has 52160 tweets
