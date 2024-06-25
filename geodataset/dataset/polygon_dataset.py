import time
from pathlib import Path
from typing import List

import albumentations
import cv2
import numpy as np
import rasterio
from scipy.spatial import distance_matrix
from tqdm import tqdm

from geodataset.dataset.base_dataset import BaseLabeledCocoDataset
from geodataset.utils import decode_rle_to_polygon, rle_segmentation_to_mask


class SiameseSamplerInternalDataset(BaseLabeledCocoDataset):
    def __init__(self,
                 fold: str,
                 root_path: Path or List[Path]):
        super().__init__(fold=fold, root_path=root_path, transform=None)

        self.tiles_per_category_id = self._generate_tiles_per_class()
        self._generate_tiles_base_features_vectors()

    def _generate_tiles_per_class(self):
        tiles_per_class = {}
        for idx in self.tiles:
            assert len(self.tiles[idx][
                           'labels']) == 1, "SiameseSamplerInternalDataset dataset should have only one label (polygon) per tile."
            category_id = self.tiles[idx]['labels'][0]['category_id']
            if category_id not in tiles_per_class:
                tiles_per_class[category_id] = []
            tiles_per_class[category_id].append(idx)
        return tiles_per_class

    def _generate_tiles_base_features_vectors(self):
        for idx in tqdm(self.tiles, desc="Generating base features vectors"):
            tile_path = self.tiles[idx]['path']
            bbox = self.tiles[idx]['labels'][0]['bbox']
            object_width = bbox[2]
            object_height = bbox[3]
            area = self.tiles[idx]['labels'][0]['area']
            area_ratio_box = area / (object_height * object_width)
            area_ratio_image = area / (self.tiles[idx]['height'] * self.tiles[idx]['width'])

            binary_mask_rle_segmentation = self.tiles[idx]['labels'][0]['segmentation']
            binary_mask = rle_segmentation_to_mask(binary_mask_rle_segmentation)
            polygon = decode_rle_to_polygon(binary_mask_rle_segmentation)
            furthest_points_distance = self._calculate_furthest_points_distance(polygon)
            diagonal = np.sqrt(self.tiles[idx]['height'] ** 2 + self.tiles[idx]['width'] ** 2)
            furthest_points_distance_ratio = furthest_points_distance / diagonal

            with rasterio.open(tile_path) as tile_file:
                tile = tile_file.read([1, 2, 3])
                object_color_histogram = self._calculate_color_histogram(tile, binary_mask)

            object_base_features = {
                'area_ratio_box': area_ratio_box,
                'area_ratio_image': area_ratio_image,
                'furthest_points_distance_ratio': furthest_points_distance_ratio,
                'object_color_histogram': object_color_histogram
            }

            self.tiles[idx]['object_base_features'] = object_base_features

    @staticmethod
    def _calculate_furthest_points_distance(polygon):
        points = np.array(polygon.exterior.coords)
        distances = distance_matrix(points, points)
        max_distance = np.max(distances)
        return max_distance

    @staticmethod
    def _calculate_color_histogram(image, binary_mask):
        if binary_mask.dtype != np.uint8:
            binary_mask = binary_mask.astype(np.uint8)

        color = ('r', 'g', 'b')
        histogram_data = []

        image = np.transpose(image, (1, 2, 0))

        # Calculate histograms for each channel
        for i, col in enumerate(color):
            # Calculate the histogram for each channel with the mask applied
            hist = cv2.calcHist(
                [image],
                [i],
                binary_mask,
                [25],  # histSize of 25 bins instead of 256 to reduce variance and noise (luminosity...).
                [0, 256]
            )
            histogram_data.extend(hist)

        histogram_data = np.array(histogram_data).flatten()

        # Dividing the histogram data by the number of pixels in the mask so that the vector sums to 1,
        # no matter the size of the object
        normalized_histogram_data = histogram_data / (3 * np.sum(binary_mask))  # 3 for RGB

        return normalized_histogram_data

    def __getitem__(self, idx: int):
        return self.tiles[idx]

    def __len__(self):
        return len(self.tiles)


class SiameseSamplerDataset:
    POSITIVE_LABEL = 1
    NEGATIVE_LABEL = 0

    def __init__(self,
                 dataset_config: dict,
                 transform: albumentations.core.composition.Compose,
                 area_ratio_box_similarity_weight: float = 0.25,
                 area_ratio_image_similarity_weight: float = 0.25,
                 furthest_points_distance_ratio_similarity_weight: float = 0.25,
                 color_histogram_cosine_similarity_weight: float = 0.25):
        self.dataset_config = dataset_config
        self.transform = transform

        self.datasets = dataset_config['datasets']
        self.sampling_strategies = dataset_config['sampling_strategies']

        self.area_ratio_box_similarity_weight = area_ratio_box_similarity_weight
        self.area_ratio_image_similarity_weight = area_ratio_image_similarity_weight
        self.furthest_points_distance_ratio_similarity_weight = furthest_points_distance_ratio_similarity_weight
        self.color_histogram_cosine_similarity_weight = color_histogram_cosine_similarity_weight

        assert sum([self.area_ratio_box_similarity_weight,
                    self.area_ratio_image_similarity_weight,
                    self.furthest_points_distance_ratio_similarity_weight,
                    self.color_histogram_cosine_similarity_weight]) == 1, \
            (f"The sum of the weights for the sampling similarity metrics must be equal to 1."
             f"Got {self.area_ratio_box_similarity_weight}, {self.area_ratio_image_similarity_weight},"
             f"{self.furthest_points_distance_ratio_similarity_weight}, {self.color_histogram_cosine_similarity_weight}.")

        (self.all_samples_indices,
         self.base_similarity_matrices) = self._generate_index()

    def _generate_index(self):
        all_samples_indices = []
        base_similarity_matrices = {}
        for sampling_strategy in self.sampling_strategies:
            if 'dataset_2' in sampling_strategy:
                dataset_1 = self.datasets[sampling_strategy['dataset_1']]
                dataset_2 = self.datasets[sampling_strategy['dataset_2']]
                dataset_1_key = sampling_strategy['dataset_1']
                dataset_2_key = sampling_strategy['dataset_2']
            else:
                dataset_1 = self.datasets[sampling_strategy['dataset_1']]
                dataset_2 = dataset_1
                dataset_1_key = sampling_strategy['dataset_1']
                dataset_2_key = dataset_1_key

            positive_sampling_strategy_params = sampling_strategy['positive_pairs']

            base_similarity_matrices[(dataset_1_key, dataset_2_key)] = {
                'positive_strategies': {},
                'negative_strategies': {}
            }

            for positive_sampling_strategy in positive_sampling_strategy_params['strategies']:
                samples_indices, base_similarity_matrix, n_duplicated_samples, n_unused_samples = self._generate_indices_for_sampling_strategy(
                    sampling_type='positive',
                    sampling_strategy=positive_sampling_strategy,
                    total_n_samples=positive_sampling_strategy_params['total_n_samples'],
                    lower_than_percentile=positive_sampling_strategy_params['lower_than_percentile'],
                    higher_than_percentile=None,
                    dataset_1=dataset_1,
                    dataset_2=dataset_2,
                    dataset_1_key=dataset_1_key,
                    dataset_2_key=dataset_2_key,
                    negative_margin=None
                )
                base_similarity_matrices[(dataset_1_key, dataset_2_key)]['positive_strategies'][positive_sampling_strategy] = base_similarity_matrix
                all_samples_indices.extend(samples_indices)
                print(f"Number of duplicated samples: {n_duplicated_samples}."
                      f" Number of unused samples: {n_unused_samples}."
                      f" Total number of samples: {len(samples_indices)}.")

            for negative_sampling_strategy in sampling_strategy['negative_pairs']['strategies']:
                samples_indices, base_similarity_matrix, n_duplicated_samples, n_unused_samples = self._generate_indices_for_sampling_strategy(
                    sampling_type='negative',
                    sampling_strategy=negative_sampling_strategy,
                    total_n_samples=sampling_strategy['negative_pairs']['total_n_samples'],
                    lower_than_percentile=None,
                    higher_than_percentile=sampling_strategy['negative_pairs']['higher_than_percentile'],
                    dataset_1=dataset_1,
                    dataset_2=dataset_2,
                    dataset_1_key=dataset_1_key,
                    dataset_2_key=dataset_2_key,
                    negative_margin=sampling_strategy['negative_pairs']['margin']
                )
                base_similarity_matrices[(dataset_1_key, dataset_2_key)]['negative_strategies'][negative_sampling_strategy] = base_similarity_matrix
                all_samples_indices.extend(samples_indices)
                print(f"Number of duplicated samples: {n_duplicated_samples}."
                      f" Number of unused samples: {n_unused_samples}."
                      f" Total number of samples: {len(samples_indices)}.")

            return all_samples_indices, base_similarity_matrices

    def _generate_indices_for_sampling_strategy(self,
                                                sampling_type: str,
                                                sampling_strategy: str,
                                                total_n_samples: int,
                                                lower_than_percentile: float or None,
                                                higher_than_percentile: float or None,
                                                dataset_1: SiameseSamplerInternalDataset,
                                                dataset_2: SiameseSamplerInternalDataset,
                                                dataset_1_key: str,
                                                dataset_2_key: str,
                                                negative_margin: int or None):

        print(f"Generating indices for sampling strategy '{sampling_type}-{sampling_strategy}'"
              f" for datasets '{dataset_1_key}' and '{dataset_2_key}'...")

        assert sampling_type in ['positive', 'negative'],\
            f"sampling_type must be either 'positive' or 'negative', got {sampling_type}"

        if sampling_type == 'positive':
            assert 0 <= lower_than_percentile <= 100, "lower_than_percentile must be an integer in 0 < p <= 100"
        if sampling_type == 'negative':
            assert 0 <= higher_than_percentile <= 100, "higher_than_percentile must be an integer in 0 < p <= 100"
            assert negative_margin is not None and negative_margin > 0, \
                f"negative_margin must be provided for negative sampling and must be a positive value. Got {negative_margin}."

        similarity_matrix = self._generate_base_similarity_matrix(dataset_1.tiles, dataset_2.tiles)

        samples_indices = []
        n_duplicated_samples = 0
        n_unused_samples = 0
        if sampling_strategy == 'class':
            # WARNING! expects that a given category_id in both datasets represents the same category/class
            categories_pairs = [(category_1, category_2) for category_1 in dataset_1.tiles_per_category_id
                                for category_2 in dataset_2.tiles_per_category_id
                                if category_1 is not None and category_2 is not None and category_2 <= category_1]

            if sampling_type == 'positive':
                categories_pairs = [(category_1, category_2) for category_1, category_2 in categories_pairs if category_1 == category_2]
            elif sampling_type == 'negative':
                categories_pairs = [(category_1, category_2) for category_1, category_2 in categories_pairs if category_1 != category_2]

            # Making sure that each pair of categories has the same number of samples, to get a balanced dataset
            n_samples_per_category_pair = total_n_samples // len(categories_pairs)

            for (category_1, category_2) in categories_pairs:
                category_1_idx = np.array(dataset_1.tiles_per_category_id[category_1])
                category_2_idx = np.array(dataset_2.tiles_per_category_id[category_2])

                category_similarity_matrix = similarity_matrix[np.ix_(category_1_idx, category_2_idx)]
                category_similarity_matrix_flat = category_similarity_matrix.flatten()

                if sampling_type == 'positive':
                    threshold_value = np.percentile(category_similarity_matrix_flat, lower_than_percentile)
                    indices = np.where(category_similarity_matrix <= threshold_value)
                    values = category_similarity_matrix[indices]
                    sorted_indices = np.argsort(values)   # We want the lowest values first
                else:
                    threshold_value = np.percentile(category_similarity_matrix_flat, higher_than_percentile)
                    indices = np.where(category_similarity_matrix >= threshold_value)
                    values = category_similarity_matrix[indices]
                    sorted_indices = np.argsort(values)
                    sorted_indices = sorted_indices[::-1]  # Reverse the order to get the highest values first

                sorted_2d_indices = [(indices[0][i], indices[1][i]) for i in sorted_indices]

                # Finding the original indices
                original_indices = [(category_1_idx[i], category_2_idx[j]) for i, j in sorted_2d_indices]

                # Making sure we have the best n_samples_per_category_pair, even if it means duplicating some of them.
                repeat_count = (n_samples_per_category_pair // len(original_indices)) + 1
                extended_list = original_indices * repeat_count
                final_list = extended_list[:n_samples_per_category_pair]

                n_duplicated_samples += max(0, len(final_list) - len(original_indices))
                n_unused_samples += max(0, len(original_indices) - len(final_list))

                for (idx_1, idx_2) in final_list:
                    if sampling_type == 'positive':
                        label = self.POSITIVE_LABEL
                        margin = 0
                    else:
                        label = self.NEGATIVE_LABEL
                        margin = negative_margin

                    samples_indices.append((dataset_1_key, idx_1, dataset_2_key, idx_2, label, margin))

        elif sampling_strategy == 'any':
            b = 1  # TODO
        elif sampling_strategy == 'none':
            c = 2  # TODO

        return samples_indices, similarity_matrix, n_duplicated_samples, n_unused_samples

    def _generate_base_similarity_matrix(self, tiles_1: dict, tiles_2: dict):
        area_ratio_box_similarity_matrix = self._compute_single_values_similarities(
            values_1=np.array([tile['object_base_features']['area_ratio_box'] for tile in tiles_1.values()]),
            values_2=np.array([tile['object_base_features']['area_ratio_box'] for tile in tiles_2.values()])
        )

        area_ratio_image_similarities = self._compute_single_values_similarities(
            values_1=np.array([tile['object_base_features']['area_ratio_image'] for tile in tiles_1.values()]),
            values_2=np.array([tile['object_base_features']['area_ratio_image'] for tile in tiles_2.values()])
        )

        furthest_points_distance_ratio_similarities = self._compute_single_values_similarities(
            values_1=np.array([tile['object_base_features']['furthest_points_distance_ratio'] for tile in tiles_1.values()]),
            values_2=np.array([tile['object_base_features']['furthest_points_distance_ratio'] for tile in tiles_2.values()])
        )

        stacked_color_histograms_1 = np.stack([tile['object_base_features']['object_color_histogram'] for tile in tiles_1.values()], axis=0)
        stacked_color_histograms_2 = np.stack([tile['object_base_features']['object_color_histogram'] for tile in tiles_2.values()], axis=0)
        norms_1 = np.linalg.norm(stacked_color_histograms_1, axis=1, keepdims=True)  # Calculate the L2 norms of the histograms
        norms_2 = np.linalg.norm(stacked_color_histograms_2, axis=1, keepdims=True)
        normalized_stacked_color_histograms_1 = stacked_color_histograms_1 / norms_1  # Normalize each histogram to have a unit length
        normalized_stacked_color_histograms_2 = stacked_color_histograms_2 / norms_2
        color_histogram_cosine_similarities = np.dot(normalized_stacked_color_histograms_1, normalized_stacked_color_histograms_2.T)

        base_similarity_matrix = (self.area_ratio_box_similarity_weight * area_ratio_box_similarity_matrix +
                                  self.area_ratio_image_similarity_weight * area_ratio_image_similarities +
                                  self.furthest_points_distance_ratio_similarity_weight * furthest_points_distance_ratio_similarities +
                                  self.color_histogram_cosine_similarity_weight * color_histogram_cosine_similarities)

        return base_similarity_matrix

    @staticmethod
    def _compute_single_values_similarities(values_1: np.ndarray, values_2: np.ndarray):
        values_1 = values_1.reshape(-1, 1)
        values_2 = values_2.reshape(-1, 1)
        pairwise_diff = np.abs(values_1 - values_2.T)
        similarities = 1 - pairwise_diff
        return similarities

    def __len__(self):
        return len(self.all_samples_indices)

    def __getitem__(self, idx):
        dataset_1_key, idx_1, dataset_2_key, idx_2, label, margin = self.all_samples_indices[idx]
        tile_1 = self.datasets[dataset_1_key][idx_1]
        tile_2 = self.datasets[dataset_2_key][idx_2]

        with rasterio.open(tile_1['path']) as tile_file:
            data_1 = tile_file.read([1, 2, 3])
        with rasterio.open(tile_2['path']) as tile_file:
            data_2 = tile_file.read([1, 2, 3])

        if self.transform:
            data_1 = self.transform(image=data_1.transpose((1, 2, 0)))['image'].transpose((2, 0, 1))
            data_2 = self.transform(image=data_2.transpose((1, 2, 0)))['image'].transpose((2, 0, 1))

        data_1 = data_1 / 255
        data_2 = data_2 / 255

        return data_1, data_2, label, margin


class SiameseValidationDataset(BaseLabeledCocoDataset):
    def __init__(self, fold: str, root_path: Path or List[Path]):
        super().__init__(fold, root_path)

        for idx in self.tiles:
            assert len(self.tiles[idx]['labels']) == 1, \
                "SiameseValidationDataset dataset should have exactly one annotation label (polygon) per tile."

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx: int):
        tile = self.tiles[idx]

        with rasterio.open(tile['path']) as tile_file:
            data = tile_file.read([1, 2, 3])

        label = tile['labels'][0]['category_id']

        if label is None:
            label = -1

        data = data / 255

        return data, label


