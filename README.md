# Geo Dataset

## Description

This package provide essential tools for cutting rasters and their labels into smaller tiles, useful for machine learning tasks. Also provides datasets compatible with pytorch.

## Installation

```bash
pip install git+ssh://git@github.com/hugobaudchon/geodataset.git
```

## Basic usage

### Detection
The class LabeledRasterTilerizer can tilerize a raster and its labels (.csv, .xml, .gpkg, .geojson and .shp). The COCO dataset generated with task='detection' will only contain bbox labels, even if the labels from the source file are polygons. 
#### Detection with .tif raster and .csv labels
```python
from pathlib import Path
from geodataset.tilerize import LabeledRasterTilerizer


tilerizer = LabeledRasterTilerizer(task='detection',
                                   dataset_name='CarlosVeraArteaga',
                                   raster_path=Path('Data/raw/wwf_ecuador/RGB Orthomosaics/Carlos Vera Arteaga RGB.tif'),
                                   labels_path=Path('Data/raw/wwf_ecuador/annotations/cleaned/clean_annotations.csv'),
                                   output_path=Path('Data/pre_processed/test/carlos_vera_arteaga'),
                                   scale_factor=0.2,                            # optional
                                   use_rle_for_labels=True,                     # optional
                                   min_intersection_ratio=0.9,                  # optional
                                   ignore_tiles_without_labels=False,           # optional
                                   ignore_mostly_black_or_white_tiles=True      # optional
                                   )
                                   
tilerizer.generate_coco_dataset(tile_size=1024, overlap=0.5, start_counter_tile=0)
```
### Segmentation
The class LabeledRasterTilerizer can tilerize a raster and its labels (.gpkg, .geojson and .shp). The COCO dataset generated with task='segmentation' will contain mask polygons.

#### Segmentation with .tif raster and .gpkg labels
```python
from pathlib import Path
from geodataset.tilerize import LabeledRasterTilerizer


tilerizer = LabeledRasterTilerizer(task='segmentation',
                                   dataset_name='quebectreesZ1',
                                   raster_path=Path('Data/raw/quebec_trees_dataset_2021-09-02/2021-09-02/zone1/2021-09-02-sbl-z1-rgb-cog.tif'),
                                   labels_path=Path('Data/raw/quebec_trees_dataset_2021-09-02/Z1_polygons.gpkg'),
                                   output_path=Path('Data/pre_processed/test/Z1_quebec_trees'),
                                   scale_factor=0.5,                        # optional
                                   use_rle_for_labels=True,                 # optional
                                   min_intersection_ratio=0.9,              # optional
                                   ignore_tiles_without_labels=False,       # optional
                                   ignore_mostly_black_or_white_tiles=True  # optional
                                   )
                                   
tilerizer.generate_coco_dataset(tile_size=1024, overlap=0.5, start_counter_tile=0)
```




## TODO Features needed:

1) load rasters from:
   1) .tif
   2) .png
2) load point clouds from:
   1) .las
   2) .laz
2) load labels from:
   1) .xml
   2) .csv
   3) .geojson
   4) .gpkg
   5) .shp
3) change resolution:
   1) raster
   2) point cloud
   3) labels
4) cut rasters and point cloud into tiles:
   1) tile size
   2) overlap between tiles
   3) area of interest for train, valid and/or test folds
      1) polygon (rectangle or more complex)
      2) % of tiles for different folds (dict fold_name => %)
   4) Remove mostly black or white tiles
   5) Optionally remove tiles without associated labels
5) save tiles and labels to disk in structured files
   1) COCO for labels
   2) Masks in RLE (compatibility with how pycocotools works with datasets and dataloader)
6) load tiles and labels into an iterable dataset
   1) merge datasets together and make sure tile ids are unique across all rasters
7) allow the retrieval of real world coordinates for each tile (if original raster has a CRS)
