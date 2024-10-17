Examples
========

Area of Interest
~~~~~~~~~~~~~~~~

In order to provide more flexibility when tilerizing a raster, geodataset supports areas of interest (AOI).
2 types of AOIs configs are supported: AOIGeneratorConfig and AOIFromPackageConfig.
These configs can then be passed to the tilerizers to divide the raster into groups of tiles for each AOI.

Specifying an AOI config for a Tilerizer is optional.
If no AOI config is passed to the Tilerizer, all the tiles will be kept in a single 'all' dataset.

.. code-block:: python

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
        aois={'train': 'QGIS_projects/train_aoi.gpkg',
              'valid': 'QGIS_projects/valid_aoi.shp',
              'test': 'Data/raw/quebec_trees_dataset_2021-09-02/inference_zone.gpkg'
              }
    )

    # For AOIGeneratorConfig, you can also specify additional parameters to control the generation of the AOIs,
    # where the 'actual_name' separates a same AOI 'train' into 2 parts, allowing an other aoi 'valid' in the middle.
    # The 'priority_aoi' can be used on a single aoi to force its tiles to be whole and not partially blacked-out because
    # they overlap other aois tiles (this is useful for small aois compared to others, like when "percentage" = 0.01).
    aoi_gen_config = AOIGeneratorConfig(
    aoi_type="band",  # currently supports 'band' and 'corner'
    aois={"train1": {"percentage": 0.495, "position": 1, "actual_name": "train"},
          "valid": {"percentage": 0.01, "position": 2, "priority_aoi": True},
          "train2": {"percentage": 0.495, "position": 3, "actual_name": "train"}},
    )

Unlabeled Raster
~~~~~~~~~~~~~~~~

The class RasterTilerizer can tilerize a raster, without labels. The tiles are then stored in the output_path/tiles.

.. code-block:: python

    from pathlib import Path
    from geodataset.tilerize import RasterTilerizer

    tilerizer = RasterTilerizer(
        raster_path='/Data/raw/wwf_ecuador/RGB Orthomosaics/Carlos Vera Arteaga RGB.tif',
        output_path='/Data/pre_processed/test',
        tile_size=1024,
        tile_overlap=0.5,
        aois_config=aoi_gen_config,
        ground_resolution=0.05,                          # optional, scale_factor must be None if used.
        scale_factor=0.5,                                # optional, ground_resolution must be None if used.
        ignore_black_white_alpha_tiles_threshold=0.8     # optional
    )

    tilerizer.generate_tiles()

The class RasterTilerizerGDF can tilerize a raster, without labels, and return the tiles as boxes in a GeoDataFrame. It does not output anything to the disk.

.. code-block:: python

    from pathlib import Path
    from geodataset.tilerize import RasterTilerizerGDF

    tilerizer = RasterTilerizerGDF(
        raster_path='/Data/raw/wwf_ecuador/RGB Orthomosaics/Carlos Vera Arteaga RGB.tif',
        tile_size=1024,
        tile_overlap=0.5,
        aois_config=aoi_gen_config,
        ground_resolution=0.05,                          # optional, scale_factor must be None if used.
        scale_factor=0.5,                                # optional, ground_resolution must be None if used.
        ignore_black_white_alpha_tiles_threshold=0.8     # optional
    )

    tiles_boxes_gdf = tilerizer.generate_tiles_gdf()

Labeled Raster
~~~~~~~~~~~~~~

The class LabeledRasterTilerizer can tilerize a raster and its labels (.gpkg, .geojson, .shp, .csv and .xml).

.. code-block:: python

    from pathlib import Path
    from geodataset.tilerize import LabeledRasterTilerizer

    tilerizer = LabeledRasterTilerizer(
        raster_path='Data/raw/quebec_trees_dataset_2021-09-02/2021-09-02/zone1/2021-09-02-sbl-z1-rgb-cog.tif',
        labels_path='Data/raw/quebec_trees_dataset_2021-09-02/Z1_polygons.gpkg',
        output_path='Data/pre_processed/test',
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

Dataset
~~~~~~~

Geodataset provides the DetectionLabeledRasterCocoDataset and SegmentationLabeledRasterCocoDataset classes which given a single or a list of root folder(s), will recursively go into each subdirectory and parse the COCO json files matching a specific 'fold',
and the associated images paths.

There is also a DetectionUnlabeledRasterDataset class which only loads tiles (useful for inference, where we don't have labels, or for pre-training a model in a self-supervised manner).

These classes can then be directly used with a torch Dataloader.

You can also provide an albumentation transform (optional) to the dataset classes to augment the data when training a model.

.. code-block:: python

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
        root_path=['Data/pre_processed/subset_1',
                   'Data/pre_processed/subset_2'],
        fold="train",
        transform=augment_transform
    )

    # Labeled Segmentation Dataset
    segmentation_valid_ds = SegmentationLabeledRasterCocoDataset(
        root_path='Data/pre_processed/all_datasets',
        fold="valid",
        transform=None
    )

    # Unlabeled Dataset (useful for inference or unsupervised pre-training)
    unlabeled_infer_ds = UnlabeledRasterDataset(
        root_path='Data/pre_processed/inference_data',
        fold="infer",     # assuming the tiles were tilerized using an aoi 'infer' instead of 'train', 'valid'...
        transform=None
    )


Aggregator
~~~~~~~~~~

The Aggregator class can be used to apply Non-Maximum Suppression style algorithms to aggregate bounding box or instance
segmentation predictions from a model from multiple tiles/images, and then save the results in a COCO json file.

For aggregating detection bounding boxes, you should currently use the nms_algorithm='iou' option.
For aggregating instance segmentation polygons, you can use both 'iou' and 'ioa-disambiguate', depending on what you need.


.. code-block:: python

    from shapely.geometry import box, Polygon
    from geodataset.aggregator import Aggregator

    # aggregating detection bounding boxes from a coco file on the disk:
    aggregator = Aggregator.from_coco(
        output_path='your_output_path',
        tiles_folder_path='path_to_folder_containing_tiles',
        coco_json_path='path_to_coco_json_file',
        polygons=[[box(0, 0, 1, 1), box(1, 1, 2, 2)],
                  [box(0, 0, 1, 1), box(1, 1, 2, 2)]],
        scores_names=['detection_score'],
        classes_names=['detection_class'],
        score_threshold=0.3,
        nms_threshold=0.8,
        nms_algorithm='iou'
    )

    # aggregating detection bounding boxes from in-memory polygons:
    aggregator = Aggregator.from_polygons(
        output_path='your_output_path',
        tiles_paths=['tile_1_path', 'tile_2_path'],
        polygons=[[box(0, 0, 1, 1), box(1, 1, 2, 2)],
                  [box(0, 0, 1, 1), box(1, 1, 2, 2)]],
        scores=[[0.9, 0.8],
                [0.7, 0.85]],
        classes=[[1, 2],
                 [2, 1]],
        score_threshold=0.3,
        nms_threshold=0.8,
        nms_algorithm='iou'
    )

    # aggregating instance segmentation polygons from in-memory polygons, with 2 different sets of scores
    # (you can also only use 1 set of scores if you want):
    aggregator = Aggregator.from_polygons(
        output_path='your_output_path',
        tiles_paths=['tile_1_path', 'tile_2_path'],
        polygons=[[Polygon([(0, 0), (1, 0), (0, 1)]), Polygon([(1, 1), (2, 1), (1, 2)])],
                  [Polygon([(2, 2), (3, 2), (2, 3)]), Polygon([(3, 3), (4, 3), (3, 4)])]],
        scores={'detection_score': [[0.9, 0.8],
                                    [0.7, 0.85]],
                'segmentation_score': [[0.6, 0.5],
                                       [0.9, 0.3]]},
        classes=[[1, 2],
                 [2, 1]],
        scores_weights={'detection_score': 2,
                        'segmentation_score': 1},
        score_threshold=0.3,
        nms_threshold=0.8,
        nms_algorithm='ioa-disambiguate',
        best_geom_keep_area_ratio=0.5
    )
