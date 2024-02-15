from pathlib import Path

from geodataset.tilerize import RasterDetectionTilerizer

tilerizer = RasterDetectionTilerizer(dataset_name='quebectreesZ1',
                                     raster_path=Path('/home/hugobaudchon/Documents/Data/raw/quebec_trees_dataset_2021-09-02/2021-09-02/zone1/2021-09-02-sbl-z1-rgb-cog.tif'),
                                     labels_path=Path('/home/hugobaudchon/Documents/Data/raw/quebec_trees_dataset_2021-09-02/Z1_polygons.gpkg'),
                                     output_path=Path('/home/hugobaudchon/Documents/Data/pre_processed/test/Z1_quebec_trees'),
                                     scale_factor=0.5)

tilerizer.generate_coco_dataset(tile_size=1024, overlap=0.5, start_counter_tile=0)


