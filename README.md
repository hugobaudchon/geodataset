# Geo Dataset

### Description

This package provide essential tools for cutting rasters and their labels into smaller tiles, useful for machine learning tasks. Also provides datasets compatible with pytorch.

### Installation

```bash
pip install git+ssh://git@github.com/hugobaudchon/geodataset.git
```

# Examples

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
aoi_gen_config = AOIGeneratorConfig(
    aoi_type='band',     # currently supports 'band' and 'corner'
    aois={'train': {'percentage': 0.7, 'position': 2},
          'valid': {'percentage': 0.15, 'position': 1},
          'test': {'percentage': 0.15, 'position': 3}
          }
)

# AOIs are provided as polygons in geopackages (.gpkg, .geojson or .shp)
aoi_gpkg_config = AOIFromPackageConfig(
    aois={'train': Path('QGIS_projects/train_aoi.shp'),
          'valid': Path('QGIS_projects/valid_aoi.shp'),
          'test': Path('Data/raw/quebec_trees_dataset_2021-09-02/inference_zone.gpkg')
          }
)
```

### Unlabeled Raster

The class RasterTilerizer can tilerize a raster, without labels. The tiles are then stored in the output_path/tiles.

```python
from pathlib import Path
from geodataset.tilerize import RasterTilerizer


tilerizer = RasterTilerizer(
    raster_path=Path('/Data/raw/wwf_ecuador/RGB Orthomosaics/Carlos Vera Arteaga RGB.tif'),
    output_path=Path('/Data/pre_processed/test'),
    tile_size=1024,
    tile_overlap=0.5,
    aois_config=aoi_gen_config,
    ground_resolution=0.05,                          # optional, scale_factor must be None if used.
    scale_factor=0.5,                                # optional, ground_resolution must be None if used.
    ignore_black_white_alpha_tiles_threshold=0.8     # optional
)

tilerizer.generate_tiles()
```

The class RasterTilerizerGDF can tilerize a raster, without labels, and return the tiles as boxes in a GeoDataFrame. It does not output anything to the disk.

```python
from pathlib import Path
from geodataset.tilerize import RasterTilerizerGDF


tilerizer = RasterTilerizerGDF(
    raster_path=Path('/Data/raw/wwf_ecuador/RGB Orthomosaics/Carlos Vera Arteaga RGB.tif'),
    tile_size=1024,
    tile_overlap=0.5,
    aois_config=aoi_gen_config,
    ground_resolution=0.05,                          # optional, scale_factor must be None if used.
    scale_factor=0.5,                                # optional, ground_resolution must be None if used.
    ignore_black_white_alpha_tiles_threshold=0.8     # optional
)

tiles_boxes_gdf = tilerizer.generate_tiles_gdf()
```


### Labeled Raster

The class LabeledRasterTilerizer can tilerize a raster and its labels (.gpkg, .geojson, .shp, .csv and .xml).


```python
from pathlib import Path
from geodataset.tilerize import LabeledRasterTilerizer


tilerizer = LabeledRasterTilerizer(
    raster_path=Path('Data/raw/quebec_trees_dataset_2021-09-02/2021-09-02/zone1/2021-09-02-sbl-z1-rgb-cog.tif'),
    labels_path=Path('Data/raw/quebec_trees_dataset_2021-09-02/Z1_polygons.gpkg'),
    output_path=Path('Data/pre_processed/test'),
    tile_size=1024,
    tile_overlap=0.5,
    aois_config=aoi_gpkg_config,
    ground_resolution=0.05,                          # optional, scale_factor must be None if used.
    scale_factor=0.5,                                # optional, ground_resolution must be None if used.
    use_rle_for_labels=True,                         # optional
    min_intersection_ratio=0.9,                      # optional
    ignore_tiles_without_labels=False,               # optional
    ignore_black_white_alpha_tiles_threshold=0.8,    # optional
    main_label_category_column_name='Label',         # optional
    other_labels_attributes_column_names=None        # optional
)
                                   
tilerizer.generate_coco_dataset()
```

### Dataset

Geodataset provides the DetectionLabeledRasterCocoDataset and SegmentationLabeledRasterCocoDataset classes which given a single or a list of root folder(s), will recursively go into each subdirectory and parse the COCO json files matching a specific 'fold',
and the associated images paths.

There is also a DetectionUnlabeledRasterDataset class which only loads tiles (useful for inference, where we don't have labels, or for pre-training a model in a self-supervised manner).

These classes can then be directly used with a torch Dataloader.

You can also provide an albumentation transform (optional) to the dataset classes to augment the data when training a model. 

```python
from pathlib import Path
from geodataset.dataset import DetectionLabeledRasterCocoDataset, SegmentationLabeledRasterCocoDataset, UnlabeledRasterDataset
import albumentations as A

augment_transform = A.Compose([
            A.HorizontalFlip(),
            A.VerticalFlip(),
        ],
            bbox_params=A.BboxParams(
                format='pascal_voc',
                label_fields=['labels'],
                min_area=0.,
                min_visibility=0.,
            ))

# Labeled Detection Dataset
detection_train_ds = DetectionLabeledRasterCocoDataset(
    root_path=Path('Data/pre_processed/all_datasets'),
    fold="train",
    transform=augment_transform
)

# Labeled Segmentation Dataset
segmentation_valid_ds = SegmentationLabeledRasterCocoDataset(
    root_path=Path('Data/pre_processed/all_datasets'),
    fold="valid",
    transform=None
)

# Unlabeled Dataset
unlabeled_infer_ds = UnlabeledRasterDataset(
    root_path=Path('Data/pre_processed/inference_data'),
    fold="infer",     # assuming the tiles were tilerized using an aoi 'infer' instead of 'train', 'valid'...
    transform=None
)
                                   

```