from tqdm import tqdm
import numpy as np
import laspy
import rasterio
import pickle
import xmltodict
from pathlib import Path
from PIL import Image
import einops
import pandas as pd
import geopandas as gpd

from shapely.geometry import box


class Tilerizer:
    def __init__(self,
                 data_path: Path,
                 annot_path: Path = None,
                 task: str = 'detection',
                 modality: str = 'image'):
        self.data_path = data_path
        self.data_extension = self.data_path.suffix
        self.annot_path = annot_path
        self.task = task
        self.modality = modality
        self.metadata, self.data = self._load_data()
        if self.annot_path:
            self.labels, self.categories, self.agb = self._load_annots()
        if self.data is None:
            raise Exception('Data is None, problem occurred during loading.')

    def _load_data(self):
        if self.modality == 'image':
            if self.data_extension == '.png':
                with Image.open(self.data_path) as img:
                    data = np.array(img)
                    data = einops.rearrange(data, 'h w c -> c h w')
                metadata = None
            elif self.data_extension == '.tif':
                with rasterio.open(self.data_path) as src:
                    metadata = src.profile
                    data = src.read()
            else:
                raise Exception('Data format {} not supported yet.'.format(self.data_extension))
        elif self.modality == 'point_cloud':
            try:
                with laspy.open(self.data_path) as pc_file:
                    data = pc_file.read()
                metadata = None
            except Exception:
                return None, None
        else:
            raise Exception('Modality {} is not supported yet.'.format(self.modality))
        return metadata, data

    def _load_annots(self):
        agb = None
        if self.task == 'detection':
            if self.annot_path.suffix.lower() == '.xml':
                with open(self.annot_path, 'r') as annotation_file:
                    annotation = xmltodict.parse(annotation_file.read())
                labels = []
                if isinstance(annotation['annotation']['object'], list):
                    for bbox in annotation['annotation']['object']:
                        xmin = bbox['bndbox']['xmin']
                        ymin = bbox['bndbox']['ymin']
                        xmax = bbox['bndbox']['xmax']
                        ymax = bbox['bndbox']['ymax']
                        labels.append([float(xmin), float(ymin), float(xmax), float(ymax)])
                else:
                    xmin = annotation['annotation']['object']['bndbox']['xmin']
                    ymin = annotation['annotation']['object']['bndbox']['ymin']
                    xmax = annotation['annotation']['object']['bndbox']['xmax']
                    ymax = annotation['annotation']['object']['bndbox']['ymax']
                    labels.append([float(xmin), float(ymin), float(xmax), float(ymax)])
                categories = None
                agb = None
            elif self.annot_path.suffix == '.csv':
                annots = pd.read_csv(self.annot_path)

                file_name_prefix = self.data_path.name.split('.')[-2]

                if 'img_name' in annots and 'Xmin' in annots and file_name_prefix in set(annots['img_name']):
                    annots = annots[annots['img_name'] == file_name_prefix]
                    labels = annots[['Xmin', 'Ymin', 'Xmax', 'Ymax']].values.tolist()
                else:
                    annots = annots[annots['img_path'] == self.data_path.name]
                    labels = annots[['xmin', 'ymin', 'xmax', 'ymax']].values.tolist()

                if 'group' in annots.columns:
                    categories = annots['group'].to_numpy()
                else:
                    categories = None
                if 'AGB' in annots.columns:
                    agb = annots['AGB'].to_numpy()
                else:
                    categories = None
            elif self.annot_path.suffix in ['.geojson', '.gpkg', '.shp']:
                labels = self._load_polygons_labels_and_convert_to_pixels(
                    polygon_file_path=self.annot_path,
                    tif_file_path=self.data_path
                )
                categories = None
            else:
                raise Exception('Annotation format {} not supported yet.'.format(self.annot_path.suffix))
        else:
            raise Exception('Task {} not implemented yet.'.format(self.task))

        return labels, categories, agb

    @staticmethod
    def _load_polygons_labels_and_convert_to_pixels(polygon_file_path: Path, tif_file_path: Path):
        """
        Load polygons from a GeoJSON or GPKG file, find their bounding box, and convert to pixel coordinates using a TIFF file.

        Parameters:
        - polygon_file_path: Path to the GeoJSON or GPKG file containing polygons.
        - tif_file_path: Path to a TIFF file for extracting CRS and converting coordinates to pixels.

        Returns:
        - A GeoDataFrame with polygons converted to pixel coordinates based on the TIFF file's CRS.
        """

        # Load polygons
        if polygon_file_path.suffix in ['.geojson', '.gpkg', '.shp']:
            polygons = gpd.read_file(polygon_file_path)
        else:
            raise ValueError("Unsupported file format for polygons. Please use GeoJSON (.geojson), GPKG (.gpkg) or Shapefile (.shp).")

        # Load TIFF file and check CRS
        with rasterio.open(tif_file_path) as src:
            if src.crs is None:
                raise ValueError("The TIFF file does not contain a CRS.")
            tif_crs = src.crs
            transform = src.transform

        # Convert polygons to the same CRS as the TIFF
        if polygons.crs != tif_crs:
            polygons = polygons.to_crs(tif_crs)

        # Calculate bounding box for each polygon and convert to pixel coordinates
        def get_pixel_bbox(geom):
            minx, miny, maxx, maxy = geom.bounds
            # Convert the bounding box corners to pixel coordinates
            top_left = ~transform * (minx, maxy)
            bottom_right = ~transform * (maxx, miny)
            return [top_left[0], bottom_right[1], bottom_right[0], top_left[1]]

        labels_pixel_bounds = polygons.geometry.apply(get_pixel_bbox).tolist()

        return labels_pixel_bounds

    def _crop_labels(self, crop_coords):
        if self.task == 'detection':
            x_min, y_min, x_max, y_max = crop_coords
            bbox_polys = np.array([box(*bbox) for bbox in self.labels])
            crop_poly = box(*crop_coords)
            intersect_bboxes = np.array([bbox.intersection(crop_poly).area / bbox.area for bbox in bbox_polys])
            idx_full_bboxes = np.where(intersect_bboxes == 1.)[0]
            idx_partial_boxes = np.where((intersect_bboxes > 0.) & (intersect_bboxes < 1.))[0]
            # Crop boxes with partial intersection
            subboxes = np.array([bbox.intersection(crop_poly) for bbox in bbox_polys[idx_partial_boxes]])
            # Check if all boxes are in the crop
            for bbox in subboxes:
                assert bbox.intersection(crop_poly).area / bbox.area == 1., 'A box has not been cropped well!'
            valid_boxes = np.hstack((bbox_polys[idx_full_bboxes], subboxes))
            # From polygones to boxes
            valid_boxes = [list(box_poly.bounds) for box_poly in valid_boxes]
            norm_valid_boxes = [[valid_box[0] - x_min, valid_box[1] - y_min, valid_box[2] - x_min, valid_box[3] - y_min]
                                for valid_box in valid_boxes]
            idx_boxes = np.hstack((idx_full_bboxes, idx_partial_boxes))
            assert len(norm_valid_boxes) == len(idx_boxes), 'List of boxes and list of idx have different num of elements.'
            return norm_valid_boxes, idx_boxes
        else:
            raise Exception('Task {} not implemented yet.'.format(self.task))

    def create_tiles(self,
                     output_folder: Path,
                     tile_size: int,
                     tile_overlap: float,
                     area_of_interest: box or None):

        tiles_output_folder = output_folder / "tiles"
        labels_output_folder = output_folder / "labels"
        tiles_output_folder.mkdir(parents=True, exist_ok=True)
        labels_output_folder.mkdir(parents=True, exist_ok=True)

        num_rows = self.data.shape[1]
        num_cols = self.data.shape[2]
        if area_of_interest:
            bounds = area_of_interest.bounds
            for bound in bounds:
                assert type(bound) is float and 0 <= bound <= 1, \
                    ("The area_of_interest must be a box with float numbers (0 <= bound <= 1) representing the fraction "
                     "of the image we want to tilerize.")
            row_start = int(num_rows * bounds[0])
            row_end = int(num_rows * bounds[2])
            col_start = int(num_cols * bounds[1])
            col_end = int(num_cols * bounds[3])
        else:
            row_start = 0
            row_end = num_rows
            col_start = 0
            col_end = num_cols

        print('Full area size: ', self.data.shape[1:])
        print('Area of interest size: ', (row_end - row_start, col_end - col_start))
        print('Desired tile size: ', tile_size)
        print('Saving tiles...')

        samples = []
        for row in tqdm(range(row_start, row_end, int((1 - tile_overlap) * tile_size))):
            for col in tqdm(range(col_start, col_end, int((1 - tile_overlap) * tile_size))):
                window = rasterio.windows.Window(col, row, tile_size, tile_size)
                tile = self.data[:, window.row_off:window.row_off + window.height,
                       window.col_off:window.col_off + window.width]

                # If it's >50% black pixels or white pixels, just continue. No point segmenting it.
                if np.sum(tile == 0) / (tile_size * tile_size * self.data.shape[0]) > 0.5:
                    continue
                if np.sum(tile == 255) / (tile_size * tile_size * self.data.shape[0]) > 0.5:
                    continue

                if self.metadata:
                    tile_profile = self.metadata.copy()
                    tile_profile.update({
                        'height': tile.shape[1],
                        'width': tile.shape[2],
                        'transform': rasterio.windows.transform(window, self.metadata['transform'])
                    })
                else:
                    tile_profile = None

                tile_file_name = 'tile_{}_{}{}'.format(row, col, self.data_extension)
                self._save_tile(tile=tile, tile_profile=tile_profile, tile_file_name=tile_file_name,
                                output_folder=tiles_output_folder)

                crop_coords = [window.col_off, window.row_off, window.col_off + window.width,
                               window.row_off + window.height]
                labels, idx_boxes = self._crop_labels(crop_coords)
                if self.categories is not None:
                    labels = {'boxes': labels,
                              'categories': self.categories[idx_boxes]}
                    if self.agb is not None:
                        labels['AGB'] = self.agb[idx_boxes]

                label_file_name = 'labels_{}_{}.pkl'.format(row, col)
                self._save_pickle(data=labels, output_file_name=label_file_name, output_folder=labels_output_folder)

                sample = {'image': tiles_output_folder / tile_file_name,
                          'labels': labels_output_folder / label_file_name}
                samples.append(sample)

        paths_file_name = str(output_folder.name) + '_paths.pkl'
        self._save_pickle(data=samples, output_file_name=paths_file_name, output_folder=output_folder)

        return samples

    def _save_tile(self,
                   tile: np.ndarray,
                   tile_profile: dict,
                   tile_file_name: str,
                   output_folder: Path):

        output_path = output_folder / tile_file_name
        if self.data_extension == '.tif':
            with rasterio.open(output_path, 'w', **tile_profile) as dst:
                dst.write(tile)
        elif self.data_extension == '.png':
            tile = einops.rearrange(tile, 'c h w -> h w c')
            tile_im = Image.fromarray(tile)
            tile_im.save(output_path)

    @staticmethod
    def _save_pickle(data, output_folder: Path, output_file_name: str):
        output_path = output_folder / output_file_name
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)

    def create_subpointcloud(self, nb_max_points=100000):
        if self.data is None:
            print('*****')
            print('Cannot create sub point cloud, the following file does not exists: {}'.format(self.data_path))
            print('*****')
            return None
        x_window, y_window = self._define_pc_window(nb_max_points)
        num_rows = int(np.ceil(self.data.x.max() - self.data.x.min()))
        num_cols = int(np.ceil(self.data.y.max() - self.data.y.min()))
        samples = []
        print('Full area size: ({}, {})'.format(num_rows, num_cols))
        print('Desired x window size: ', x_window)
        print('Desired y window size: ', y_window)

        print('Saving sub point clouds')
        for row in tqdm(range(0, num_rows, x_window)):
            for col in tqdm(range(0, num_cols, y_window)):
                window = rasterio.windows.Window(col, row, y_window, x_window)
                x_min = self.data.x.min() + window.row_off
                y_min = self.data.y.min() + window.col_off
                sub_pc = self.data[np.where((self.data.x >= x_min) &
                                            (self.data.x < x_min + window.height) &
                                            (self.data.y >= y_min) &
                                            (self.data.y < y_min + window.width))[0]]

                sub_pc_name = 'sub_pc_{}_{}.laz'.format(row, col)
                try:
                    sub_pc.write(self.pcs_folder / sub_pc_name)
                    sample = {'sub_pc': self.pcs_folder / sub_pc_name}
                    samples.append(sample)
                except AttributeError:
                    # Empty point cloud
                    print('Try to save empty point cloud, skipped')
                    pass

        paths_file_name = self.data_path.stem + '_paths.pkl'
        with open(self.data_path.parent / paths_file_name, 'wb') as f:
            pickle.dump(samples, f)
        return samples

    def _define_pc_window(self, nb_max_points):
        total_nb_points = len(self.data.x)
        x_min = self.data.x.min()
        x_max = self.data.x.max()
        y_min = self.data.y.min()
        y_max = self.data.y.max()
        x_dist = 1
        y_dist = 1
        if total_nb_points <= nb_max_points:
            x_dist = int(np.ceil(x_max - x_min))
            y_dist = int(np.ceil(y_max - y_min))
            return x_dist, y_dist
        while len(np.where((self.data.x <= x_min + x_dist) & (self.data.y <= y_min + y_dist))[0]) <= nb_max_points:
            x_dist += 1
            y_dist += 1
        # remove the last iteration
        x_dist -= 1
        y_dist -= 1
        return x_dist, y_dist


if __name__ == '__main__':
    # neontree_path = Paths().get()['neontree']
    # # test_image(neontree_path)
    # test_pointcloud(neontree_path)
    # import ipdb; ipdb.set_trace()

    tilerizer = Tilerizer(
        data_path=Path("/home/hugobaudchon/Documents/Data/raw/quebec_trees_dataset_2021-09-02/2021-09-02/zone3/2021-09-02-sbl-z3-rgb-cog.tif"),
        annot_path=Path("/home/hugobaudchon/Documents/Data/raw/quebec_trees_dataset_2021-09-02/Z3_polygons.gpkg"),
    )
    tilerizer.create_tiles(output_folder=Path("/home/hugobaudchon/Documents/Data/pre_processed/quebec_trees/Z3_quebec_trees_test"),
                           tile_size=1024,
                           tile_overlap=0.5,
                           area_of_interest=box(0.5, 0, 1, 1))
