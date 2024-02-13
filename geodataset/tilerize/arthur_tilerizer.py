from contextlib import contextmanager
import numpy as np
import pandas as pd
import rasterio
from rasterio import MemoryFile
from rasterio.enums import Resampling
from pathlib import Path


class SimpleTilerizer:

    def __init__(self, tif_path, dataset_name, tiles_path, scale_factor=1):
        """
        tif_path : Pathlib
            orthomosaic path
        dataset_name: str
            dataset name
        tiles_path: Pathlib
            path to folder where to save the image tiles
        scale_factor: int
            Rescaling the tif (change pixel resolution)
        """
        self.tif_path = tif_path
        self.dataset_name = dataset_name
        self.tiles_path = Path(tiles_path)
        self.scale_factor = scale_factor
        self._create_folders()

    @contextmanager
    def _resample_raster(self, data):
        # resample data to target shape
        dataset = data.read(
            out_shape=(data.count,
                       int(data.height * self.scale_factor),
                       int(data.width * self.scale_factor)),
            resampling=Resampling.bilinear)
        # scale image transform
        transform = data.transform * data.transform.scale(
            (data.width / dataset.shape[-1]),
            (data.height / dataset.shape[-2]))

        profile = data.profile
        profile.update(transform=transform, driver='GTiff',
                       height=dataset.shape[-2], width=dataset.shape[-1])

        with MemoryFile() as memfile:
            with memfile.open(**profile) as raster_dataset: # Open as DatasetWriter
                raster_dataset.write(dataset)
                del dataset
            with memfile.open() as dataset:  # Reopen as DatasetReader
                yield dataset

    def _create_folders(self):
        self.tiles_path = self.tiles_path / 'tiles'
        self.tiles_path.mkdir(parents=True, exist_ok=True)

    def _write_paths(self, sample_paths):
        dataset_paths = pd.DataFrame(sample_paths, columns=['paths'])
        file_name = self.dataset_name + '_paths.csv'
        dataset_paths.to_csv(self.tiles_path.parent / file_name, index=False)

    def create_tiles(self, tile_size=1024, overlap=0, start_counter_tile=0):
        """
        tile_size: int
        overlap: float
        start counter tile: int
            we want all the tiles to have a unique id.
            So when applying to severa orthomosaics, we might want to set the first
            id to be 1 + the last of the previous orthomosaic
        """
        samples = []
        with rasterio.open(self.tif_path, "r") as data:
            with self._resample_raster(data) as dataset:
                # we must include the raster opening in this function to keep it on RAM
                n_x = dataset.width
                n_y = dataset.height
                print('Desired tile size: ', tile_size)
                tile_id = start_counter_tile
                print('Saving tiles')
                for j in range (0, n_y, int((1-overlap)*tile_size)):
                    print(f'  row {j}...')
                    for i in range(0, n_x, int((1-overlap)*tile_size)):
                        print(f'  column {i}...')
                        win_x = i
                        win_y = j
                        window = rasterio.windows.Window(win_x, win_y, tile_size, tile_size)
                        raster = dataset.read(window=window)
                        # If it's >50% black pixels or white pixels, just continue. No point segmenting it.
                        if np.sum(raster==0)/(tile_size*tile_size*dataset.count)>0.5:
                            continue
                        if np.sum(raster==255)/(tile_size*tile_size*dataset.count)>0.5:
                            continue
                        window_transform =  dataset.window_transform(window)
                        tile_name = 'tile_{}_{}_{}.tif'.format(self.dataset_name, j, i)
                        sample_path = self.tiles_path / tile_name
                        samples.append(sample_path)
                        with rasterio.open(
                                sample_path, 'w',
                                driver='GTiff',
                                height=tile_size,
                                width=tile_size,
                                count=dataset.count,
                                dtype=dataset.dtypes[0],
                                crs=dataset.crs,
                                transform=window_transform) as tile_raster:
                            tile_raster.write(raster)
                tile_id += 1
        print("Done")
        self._write_paths(samples)