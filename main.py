from pathlib import Path

from geodataset.geodata.geodata import GeoReferencedRaster
from geodataset.labels.detection_labels import DetectionLabels
from geodataset.dataset.detection_dataset import LabeledDetectionRasterDataset

data = GeoReferencedRaster(path=Path('C:/Users/Hugo/Documents/Data/quebec_trees_dataset_2021-09-02/2021-09-02/zone1/2021-09-02-sbl-z1-rgb-cog.tif'))
labels = DetectionLabels(path=Path('C:/Users/Hugo/Documents/Data/quebec_trees_dataset_2021-09-02/Z1_polygons.gpkg'),
                         associated_geo_data=data)

dataset = LabeledDetectionRasterDataset(data=data,
                                        labels=labels)

print(labels.labels)
print(data.crs)






