from pathlib import Path

from geodataset.geodata import Raster, PointCloud
from geodataset.labels import DetectionLabels
from geodataset.dataset import LabeledDetectionRasterDataset

raster = Raster(path=Path('/home/hugobaudchon/Documents/Data/raw/quebec_trees_dataset_2021-09-02/2021-09-02/zone1/2021-09-02-sbl-z1-rgb-cog.tif'),
                scale_factor=0.5)
labels = DetectionLabels(path=Path('/home/hugobaudchon/Documents/Data/raw/quebec_trees_dataset_2021-09-02/Z1_polygons.gpkg'),
                         associated_geo_data=raster)

dataset = LabeledDetectionRasterDataset(data=raster,
                                        labels=labels)

print(labels.labels)
print(raster.crs)
print(raster.data.shape)


