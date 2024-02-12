 # Geo Dataset #
 

TODO Features needed:
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
