# Geo Dataset

## Description

This package provide essential tools for cutting rasters and their labels into smaller tiles, useful for machine learning tasks. Also provides datasets compatible with pytorch.

## Installation

```bash
pip install git+ssh://git@github.com/hugobaudchon/geodataset.git
```

## Examples

### Area of Interest

In order to provide more flexibility when tilerizing a raster, geodataset supports areas of interest (AOI).
2 types of AOIs configs are supported: AOIGeneratorConfig and AOIFromPackageConfig.
These configs can then be passed to the tilerizers to divide the raster into groups of tiles for each AOI.

Specifying an AOI config for a Tilerizer is optional.
If no AOI config is passed to the Tilerizer, all the tiles will be kept in a single 'all' dataset.

```python
from pathlib import Path
from geodataset.aoi import AOIFromPackageConfig, AOIGeneratorConfig

# Automatically creates AOIs based on the percentage of tiles that should be in each AOI.
# The 'position' value can be None (random) or a unique value between 1 and n_aois, to force the AOIs to specific bands/corners.
aoi_gen_config = AOIGeneratorConfig(aoi_type='band',     # currently supports 'band' and 'corner'
                                    aois={'train': {'percentage': 0.7, 'position': 2},
                                          'valid': {'percentage': 0.15, 'position': 1},
                                          'test': {'percentage': 0.15, 'position': 3}
                                          })

# AOIs are provided as polygons in geopackages (.gpkg, .geojson or .shp)
aoi_gpkg_config = AOIFromPackageConfig(aois={
    'train': Path('QGIS_projects/train_aoi.shp'),
    'valid': Path('QGIS_projects/valid_aoi.shp'),
    'test': Path('Data/raw/quebec_trees_dataset_2021-09-02/inference_zone.gpkg')
})
```

### Unlabeled Raster

The class RasterTilerizer can tilerize a raster, without labels.

```python
from pathlib import Path
from geodataset.tilerize import RasterTilerizer


tilerizer = RasterTilerizer(dataset_name='TilesOnlyCarlosVeraArteagaRLE',
                            raster_path=Path('/Data/raw/wwf_ecuador/RGB Orthomosaics/Carlos Vera Arteaga RGB.tif'),
                            output_path=Path('/Data/pre_processed/test'),
                            scale_factor=0.2,
                            ignore_black_white_alpha_tiles_threshold=0.8   # optional
                            )

tilerizer.generate_tiles(tile_size=1024,
                         overlap=0.5,
                         aois_config=aoi_gen_config)
```

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
                                   output_path=Path('Data/pre_processed/test'),
                                   scale_factor=0.2,                                 # optional
                                   use_rle_for_labels=True,                          # optional
                                   min_intersection_ratio=0.9,                       # optional
                                   ignore_tiles_without_labels=False,                # optional
                                   ignore_black_white_alpha_tiles_threshold=0.8      # optional
                                   )
                                   
tilerizer.generate_coco_dataset(tile_size=1024,
                                overlap=0.5,
                                start_counter_tile=0,
                                aois_config=aoi_gen_config)
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
                                   output_path=Path('Data/pre_processed/test'),
                                   scale_factor=0.5,                             # optional
                                   use_rle_for_labels=True,                      # optional
                                   min_intersection_ratio=0.9,                   # optional
                                   ignore_tiles_without_labels=False,            # optional
                                   ignore_black_white_alpha_tiles_threshold=0.8  # optional
                                   )
                                   
tilerizer.generate_coco_dataset(tile_size=1024,
                                overlap=0.5,
                                start_counter_tile=0,
                                aois_config=aoi_gpkg_config)
```
