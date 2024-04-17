from pathlib import Path

from geodataset.aoi import AOIGeneratorConfig, AOIFromPackageConfig
from geodataset.tilerize import LabeledRasterTilerizer


def quebec_trees_parser(dataset_root: Path,
                        output_path: Path,
                        tile_size: int,
                        tile_overlap: float,
                        ground_resolution: float = 0.05):

    # Dataset can be found here: https://zenodo.org/records/8148479

    output_path = output_path / "quebec_trees"

    aoi_gen_config = AOIFromPackageConfig(aois={'train': Path('./aois/quebec_trees/train_aoi.geojson'),
                                                'valid': Path('./aois/quebec_trees/valid_aoi.geojson'),
                                                'test': Path('./aois/quebec_trees/inference_zone.gpkg')
                                                })

    raster_configs = [
        {'raster': dataset_root / '2021-09-02/zone1/2021-09-02-sbl-z1-rgb-cog.tif',
         'labels': dataset_root / 'Z1_polygons.gpkg',
         'aoi_config': aoi_gen_config
         },
        {'raster': dataset_root / '2021-09-02/zone2/2021-09-02-sbl-z2-rgb-cog.tif',
         'labels': dataset_root / 'Z2_polygons.gpkg',
         'aoi_config': aoi_gen_config
         },
        {'raster': dataset_root / '2021-09-02/zone3/2021-09-02-sbl-z3-rgb-cog.tif',
         'labels': dataset_root / 'Z3_polygons.gpkg',
         'aoi_config': aoi_gen_config
         }
    ]

    for raster_config in raster_configs:
        try:
            tilerizer = LabeledRasterTilerizer(
                raster_path=raster_config['raster'],
                labels_path=raster_config['labels'],
                output_path=output_path,
                tile_size=tile_size,
                tile_overlap=tile_overlap,
                aois_config=raster_config['aoi_config'],
                ground_resolution=ground_resolution,
                ignore_black_white_alpha_tiles_threshold=0.8,
                ignore_tiles_without_labels=True,
                main_label_category_column_name='Label')
            tilerizer.generate_coco_dataset()
        except Exception as e:
            print(e)
            print(f'Failed to parse the raster {raster_config["raster"]}.')


def neon_train_parser(dataset_root: Path,
                      output_path: Path,
                      tile_size: int,
                      tile_overlap: float,
                      ground_resolution: float = 0.05):

    # Dataset can be found here: https://zenodo.org/records/5914554
    # You need to download training.zip and annotations.zip.
    # For this script to work you then need to put the annotations folder into the training folder.

    output_path = output_path / "neon_trees"

    aoi_train = AOIGeneratorConfig(aoi_type='band', aois={'train': {'percentage': 1.0, 'position': 1}})
    aoi_valid = AOIGeneratorConfig(aoi_type='band', aois={'valid': {'percentage': 1.0, 'position': 1}})
    aoi_test = AOIGeneratorConfig(aoi_type='band', aois={'test': {'percentage': 1.0, 'position': 1}})

    neon_rgb_data = dataset_root / 'RGB'
    neon_rgb_annotations = dataset_root / 'annotations'

    raster_configs = [
        {'raster': neon_rgb_data / "2018_BART_4_322000_4882000_image_crop.tif",
         'labels': neon_rgb_annotations / "2018_BART_4_322000_4882000_image_crop.xml",
         'aoi_config': aoi_train
         },
        {'raster': neon_rgb_data / "2018_HARV_5_733000_4698000_image_crop.tif",
         'labels': neon_rgb_annotations / "2018_HARV_5_733000_4698000_image_crop.xml",
         'aoi_config': aoi_train
         },
        {'raster': neon_rgb_data / "2018_JERC_4_742000_3451000_image_crop.tif",
         'labels': neon_rgb_annotations / "2018_JERC_4_742000_3451000_image_crop.xml",
         'aoi_config': aoi_train
         },
        {'raster': neon_rgb_data / "2018_MLBS_3_541000_4140000_image_crop.tif",
         'labels': neon_rgb_annotations / "2018_MLBS_3_541000_4140000_image_crop.xml",
         'aoi_config': aoi_train
         },
        {'raster': neon_rgb_data / "2018_MLBS_3_541000_4140000_image_crop2.tif",
         'labels': neon_rgb_annotations / "2018_MLBS_3_541000_4140000_image_crop2.xml",
         'aoi_config': aoi_valid
         },
        {'raster': neon_rgb_data / "2018_NIWO_2_450000_4426000_image_crop.tif",
         'labels': neon_rgb_annotations / "2018_NIWO_2_450000_4426000_image_crop.xml",
         'aoi_config': aoi_train
         },
        {'raster': neon_rgb_data / "2018_OSBS_4_405000_3286000_image.tif",
         'labels': neon_rgb_annotations / "2018_OSBS_4_405000_3286000_image.xml",
         'aoi_config': aoi_train
         },
        {'raster': neon_rgb_data / "2018_SJER_3_258000_4106000_image.tif",
         'labels': neon_rgb_annotations / "2018_SJER_3_258000_4106000_image.xml",
         'aoi_config': aoi_train
         },
        {'raster': neon_rgb_data / "2018_SJER_3_259000_4110000_image.tif",
         'labels': neon_rgb_annotations / "2018_SJER_3_259000_4110000_image.xml",
         'aoi_config': aoi_train
         },
        {'raster': neon_rgb_data / "2018_TEAK_3_315000_4094000_image_crop.tif",
         'labels': neon_rgb_annotations / "2018_TEAK_3_315000_4094000_image_crop.xml",
         'aoi_config': aoi_train
         },
        {'raster': neon_rgb_data / "2019_DELA_5_423000_3601000_image_crop.tif",
         'labels': neon_rgb_annotations / "2019_DELA_5_423000_3601000_image_crop.xml",
         'aoi_config': aoi_test
         },
        # {'raster': neon_rgb_data / "2019_DSNY_5_452000_3113000_image_crop.tif",       # rasterio says no transform in this one
        #  'labels': neon_rgb_annotations / "2019_DSNY_5_452000_3113000_image_crop.xml",
        #  'aoi_config': aoi_train
        #  },
        {'raster': neon_rgb_data / "2019_LENO_5_383000_3523000_image_crop.tif",
         'labels': neon_rgb_annotations / "2019_LENO_5_383000_3523000_image_crop.xml",
         'aoi_config': aoi_valid
         },
        {'raster': neon_rgb_data / "2019_ONAQ_2_367000_4449000_image_crop.tif",
         'labels': neon_rgb_annotations / "2019_ONAQ_2_367000_4449000_image_crop.xml",
         'aoi_config': aoi_train
         },
        {'raster': neon_rgb_data / "2019_OSBS_5_405000_3287000_image_crop.tif",
         'labels': neon_rgb_annotations / "2019_OSBS_5_405000_3287000_image_crop.xml",
         'aoi_config': aoi_train
         },
        {'raster': neon_rgb_data / "2019_OSBS_5_405000_3287000_image_crop2.tif",
         'labels': neon_rgb_annotations / "2019_OSBS_5_405000_3287000_image_crop2.xml",
         'aoi_config': aoi_train
         },
        {'raster': neon_rgb_data / "2019_SJER_4_251000_4103000_image.tif",
         'labels': neon_rgb_annotations / "2019_SJER_4_251000_4103000_image.xml",
         'aoi_config': aoi_train
         },
        {'raster': neon_rgb_data / "2019_TOOL_3_403000_7617000_image.tif",
         'labels': neon_rgb_annotations / "2019_TOOL_3_403000_7617000_image.xml",
         'aoi_config': aoi_train
         },
        # {'raster': neon_rgb_data / "2019_YELL_2_528000_4978000_image_crop2.tif",              # No CRS
        #  'labels': neon_rgb_annotations / "2019_YELL_2_528000_4978000_image_crop2.xml",
        #  'aoi_config': aoi_train
        #  },
        # {'raster': neon_rgb_data / "2019_YELL_2_541000_4977000_image_crop.tif",              # No CRS
        #  'labels': neon_rgb_annotations / "2019_YELL_2_541000_4977000_image_crop.xml",
        #  'aoi_config': aoi_train
        #  }
    ]

    for raster_config in raster_configs:
        try:
            tilerizer = LabeledRasterTilerizer(
                raster_path=raster_config['raster'],
                labels_path=raster_config['labels'],
                output_path=output_path,
                tile_size=tile_size,
                tile_overlap=tile_overlap,
                aois_config=raster_config['aoi_config'],
                ground_resolution=ground_resolution,
                ignore_black_white_alpha_tiles_threshold=0.8,
                ignore_tiles_without_labels=True,
                main_label_category_column_name=None)
            tilerizer.generate_coco_dataset()
        except Exception as e:
            print(e)
            print(f'Failed to parse the raster {raster_config["raster"]}.')


def refores_trees_parser(dataset_root: Path,
                         output_path: Path,
                         tile_size: int,
                         tile_overlap: float,
                         scale_factor: float = 0.3):

    # Dataset can be found here: https://zenodo.org/records/6813783

    output_path = output_path / "refores_trees"

    aoi_train = AOIGeneratorConfig(aoi_type='band', aois={'train': {'percentage': 1.0, 'position': 1}})
    aoi_valid = AOIGeneratorConfig(aoi_type='band', aois={'valid': {'percentage': 1.0, 'position': 1}})
    aoi_test = AOIGeneratorConfig(aoi_type='band', aois={'test': {'percentage': 1.0, 'position': 1}})

    refores_tree_data = dataset_root / 'wwf_ecuador/RGB Orthomosaics'
    refores_tree_annotations = dataset_root / 'annotations'

    raster_configs = [
        {'raster': refores_tree_data / "Carlos Vera Arteaga RGB.tif",
         'labels': refores_tree_annotations / "cleaned/clean_annotations.csv",
         'aoi_config': aoi_train
         },
        {'raster': refores_tree_data / "Carlos Vera Guevara RGB.tif",
         'labels': refores_tree_annotations / "cleaned/clean_annotations.csv",
         'aoi_config': aoi_test
         },
        {'raster': refores_tree_data / "Flora Pluas RGB.tif",
         'labels': refores_tree_annotations / "cleaned/clean_annotations.csv",
         'aoi_config': aoi_train
         },
        {'raster': refores_tree_data / "Leonor Aspiazu RGB.tif",
         'labels': refores_tree_annotations / "cleaned/clean_annotations.csv",
         'aoi_config': aoi_valid
         },
        {'raster': refores_tree_data / "Manuel Macias RGB.tif",
         'labels': refores_tree_annotations / "cleaned/clean_annotations.csv",
         'aoi_config': aoi_train
         },
        {'raster': refores_tree_data / "Nestor Macias RGB.tif",
         'labels': refores_tree_annotations / "cleaned/clean_annotations.csv",
         'aoi_config': aoi_train
         },
    ]

    for raster_config in raster_configs:
        try:
            tilerizer = LabeledRasterTilerizer(
                raster_path=raster_config['raster'],
                labels_path=raster_config['labels'],
                output_path=output_path,
                tile_size=tile_size,
                tile_overlap=tile_overlap,
                aois_config=raster_config['aoi_config'],
                ground_resolution=None,
                scale_factor=scale_factor,
                ignore_black_white_alpha_tiles_threshold=0.8,
                ignore_tiles_without_labels=True,
                main_label_category_column_name='is_banana')
            tilerizer.generate_coco_dataset()
        except Exception as e:
            print(e)
            print(f'Failed to parse the raster {raster_config["raster"]}.')


def savanna_trees_parser(dataset_root: Path,
                         output_path: Path,
                         tile_size: int,
                         tile_overlap: float,
                         ground_resolution: float = 0.05):

    # Dataset can be found here: https://zenodo.org/records/7094916
    # You need to download the file Savanna Tree Mosaic and labels.rar

    output_path = output_path / "savanna_trees"

    aoi_train = AOIGeneratorConfig(aoi_type='band', aois={'train': {'percentage': 1.0, 'position': 1}})
    aoi_valid = AOIGeneratorConfig(aoi_type='band', aois={'valid': {'percentage': 1.0, 'position': 1}})
    aoi_test = AOIGeneratorConfig(aoi_type='band', aois={'test': {'percentage': 1.0, 'position': 1}})

    savanna_tree_data = dataset_root / 'Savanna Tree Mosaic and labels'
    savanna_tree_annotations = dataset_root / 'Savanna Tree Mosaic and labels'

    raster_configs = [
        {'raster': savanna_tree_data / "PlotS1_20210412_RGB.tif",
         'labels': savanna_tree_annotations / "PlotS1_20210412_Labels.shp",
         'aoi_config': aoi_train
         },
        {'raster': savanna_tree_data / "PlotS2_20210621_RGB.tif",
         'labels': savanna_tree_annotations / "PlotS2_20210621_Labels.shp",
         'aoi_config': aoi_train
         },
        {'raster': savanna_tree_data / "PlotS3_20210622_RGB.tif",
         'labels': savanna_tree_annotations / "PlotS3_20210622_Labels.shp",
         'aoi_config': aoi_train
         },
        {'raster': savanna_tree_data / "PlotS4_20210614_RGB.tif",
         'labels': savanna_tree_annotations / "PlotS4_20210614_Labels.shp",
         'aoi_config': aoi_test
         },
        {'raster': savanna_tree_data / "PlotS5_20210615_RGB.tif",
         'labels': savanna_tree_annotations / "PlotS5_20210615_Labels.shp",
         'aoi_config': aoi_train
         },
        {'raster': savanna_tree_data / "PlotS6_20210621_RGB.tif",
         'labels': savanna_tree_annotations / "PlotS6_20210621_Labels.shp",
         'aoi_config': aoi_train
         },
        {'raster': savanna_tree_data / "PlotS7_20210617_RGB.tif",
         'labels': savanna_tree_annotations / "PlotS7_20210617_Labels.shp",
         'aoi_config': aoi_valid
         }
    ]

    for raster_config in raster_configs:
        try:
            tilerizer = LabeledRasterTilerizer(
                raster_path=raster_config['raster'],
                labels_path=raster_config['labels'],
                output_path=output_path,
                tile_size=tile_size,
                tile_overlap=tile_overlap,
                aois_config=raster_config['aoi_config'],
                ground_resolution=ground_resolution,
                ignore_black_white_alpha_tiles_threshold=0.8,
                ignore_tiles_without_labels=True,
                main_label_category_column_name='Code')
            tilerizer.generate_coco_dataset()
        except Exception as e:
            print(e)
            print(f'Failed to parse the raster {raster_config["raster"]}.')


if __name__ == "__main__":
    quebec_trees_parser(dataset_root=Path('/home/hugobaudchon/Documents/Data/raw/quebec_trees_dataset_2021-09-02'),
                        output_path=Path('/home/hugobaudchon/Documents/Data/pre_processed/all_datasets'),
                        ground_resolution=0.05,
                        tile_size=1024,
                        tile_overlap=0.5)

    # neon_train_parser(dataset_root=Path('/home/hugobaudchon/Documents/Data/raw/NeonTreeEvaluation/training'),
    #                   output_path=Path('/home/hugobaudchon/Documents/Data/pre_processed/all_datasets'),
    #                   ground_resolution=0.05,
    #                   tile_size=1024,
    #                   tile_overlap=0.5)
    #
    # refores_trees_parser(dataset_root=Path('/home/hugobaudchon/Documents/Data/raw/wwf_ecuador'),
    #                      output_path=Path('/home/hugobaudchon/Documents/Data/pre_processed/all_datasets'),
    #                      scale_factor=0.3,      # Can't use ground_resolution here as the rasters CRS are in degree,
    #                                             # not in meter, and changing CRS un-aligns the rasters with their
    #                                             # labels by a few tens of meters.
    #                      tile_size=1024,
    #                      tile_overlap=0.5)

    # Savanna Tree Dataset is not ideal for tilerizing a detection dataset as only about 30% of the trees in each raster has labels.
    # Training a detection model on this data is really noisy.
    # savanna_trees_parser(dataset_root=Path('C:/Users/Hugo/Documents/Data/raw/Savanna Tree Mosaic and labels'),
    #                      output_path=Path('C:/Users/Hugo/Documents/Data/pre_processed/all_datasets'),
    #                      ground_resolution=0.05,
    #                      tile_size=1024,
    #                      tile_overlap=0.5)
