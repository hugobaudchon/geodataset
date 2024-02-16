from pathlib import Path

from geodataset.tilerize import RasterDetectionTilerizer

tilerizer = RasterDetectionTilerizer(dataset_name='quebectreesZ1',
                                     raster_path=Path('/home/hugobaudchon/Documents/Data/raw/quebec_trees_dataset_2021-09-02/2021-09-02/zone1/2021-09-02-sbl-z1-rgb-cog.tif'),
                                     labels_path=Path('/home/hugobaudchon/Documents/Data/raw/quebec_trees_dataset_2021-09-02/Z1_polygons.gpkg'),
                                     output_path=Path('/home/hugobaudchon/Documents/Data/pre_processed/test/Z1_quebec_trees'),
                                     scale_factor=0.5,
                                   task='segmentation')

# tilerizer = RasterLabeledTilerizer(dataset_name='neontree2018mlbs3',
#                                      raster_path=Path('/home/hugobaudchon/Documents/Data/raw/NeonTreeEvaluation/training/RGB/2018_MLBS_3_541000_4140000_image_crop.tif'),
#                                      labels_path=Path('/home/hugobaudchon/Documents/Data/raw/NeonTreeEvaluation/annotations/annotations/2018_MLBS_3_541000_4140000_image_crop.xml'),
#                                      output_path=Path('/home/hugobaudchon/Documents/Data/pre_processed/test/2018_MLBS_3_541000_4140000_image_crop'),
#                                      scale_factor=1,
#                                      task='detection')

# tilerizer = RasterLabeledTilerizer(dataset_name='CarlosVeraArteaga_seg',
#                                    raster_path=Path('/home/hugobaudchon/Documents/Data/raw/wwf_ecuador/RGB Orthomosaics/Carlos Vera Arteaga RGB.tif'),
#                                    labels_path=Path('/home/hugobaudchon/Documents/Data/raw/wwf_ecuador/annotations/cleaned/clean_annotations.csv'),
#                                    output_path=Path('/home/hugobaudchon/Documents/Data/pre_processed/test/carlos_vera_arteaga'),
#                                    scale_factor=0.2,
#                                    task='detection')

tilerizer.generate_coco_dataset(tile_size=1024, overlap=0.5, start_counter_tile=0)


