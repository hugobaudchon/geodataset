# load .env
import json
import os

import rasterio
from arcgis.gis import GIS
from dotenv import load_dotenv
import geopandas as gpd

load_dotenv()

def download_arcgis_online_labels(raster_path: str, arcgis_layer_link: str, output_gpkg_path: str):
    raster = rasterio.open(raster_path)
    raster_crs = raster.crs

    # Download labels from ArcGIS Online
    gis = GIS(os.environ.get('arcgis_online_server'),
              os.environ.get('arcgis_online_username'),
              os.environ.get('arcgis_online_password'))

    item = gis.content.get(arcgis_layer_link)
    feature_layer = item.layers[0]
    feature_set = feature_layer.query()
    feature_json = feature_set.to_geojson
    feature_json_dict = json.loads(feature_json)
    gdf = gpd.GeoDataFrame.from_features(feature_json_dict['features'], crs=raster_crs)
    gdf.to_file(output_gpkg_path, driver="GPKG")


if __name__ == "__main__":
    raster_path = '/home/hugo/Documents/Data/raw/brazil/20240130_zf2quad_m3m_rgb.cog.tif'
    arcgis_layer_link = '478ff9a9ec814314af7f7411d07b627a'
    output_gpkg_path = '/home/hugo/Documents/Data/raw/brazil/20240130_zf2quad_m3m_labels_boxes.gpkg'

    download_arcgis_online_labels(raster_path, arcgis_layer_link, output_gpkg_path)