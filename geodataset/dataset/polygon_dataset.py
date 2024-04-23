from pathlib import Path
from typing import List

import albumentations
import rasterio

from geodataset.dataset.base_dataset import BaseLabeledCocoDataset


class SiameseLabeledPolygonCocoDataset(BaseLabeledCocoDataset):
    def __init__(self,
                 fold: str,
                 root_path: Path or List[Path],
                 transform: albumentations.core.composition.Compose = None):

        super().__init__(fold=fold, root_path=root_path, transform=transform)

        for idx in self.tiles:
            assert len(self.tiles[idx]['labels']) == 1, \
                "SiameseLabeledPolygonCocoDataset dataset should have only one label per tile."

        self.pairs_indices = self._generate_pairs_indices()

        # TODO add support to balance categories (instead of going through n**2/2 pairs,
        #  go through less than that by skipping instances of the most populated classes
        #  or choosing samples with x probability for each class)

        # TODO use DINOv2 Small to provide embeddings so that I can choose the best positive and negative pairs? In terms of difficulty etc.
        # TODO =====> Or maybe instead of DINOv2, get an idea of the height/width of the objects, the area, the 'shape' (is it weird or almost a circle?) and the histogram of color distributions to choose similar/dissimilar pairs within a class or between classes (positive/negative)
        # TODO Can also sample pairs based on if the mask is from a human or from SAM (can help the model generalize better on SAM data)
        # TODO Can also use the area and perimeter of the mask to get an idea of the complexity of the polygon:
        # def calculate_complexity(contour):
        #     # Calculate area and perimeter
        #     area = cv2.contourArea(contour)
        #     perimeter = cv2.arcLength(contour, True)
        #
        #     # Circle equivalence: calculate the complexity relative to a circle
        #     circularity = (perimeter ** 2) / (4 * np.pi * area) if area > 0 else 0
        #
        #     # Simplicity metric (closer to 1 is simpler, larger numbers indicate more complexity)
        #     complexity = circularity
        #
        #     return complexity
        #
        # def find_contours(binary_image):
        #     # Find contours in the binary image
        #     contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #     return contours



    def _generate_pairs_indices(self):
        pairs_indices = []
        for i in range(len(self.tiles)):
            for j in range(i + 1, len(self.tiles)):
                pairs_indices.append((i, j))

        return pairs_indices

    def __len__(self):
        return len(self.pairs_indices)

    def __getitem__(self, idx: int):
        indices_pair = self.pairs_indices[idx]

        tile_info_1 = self.tiles[indices_pair[0]]
        tile_info_2 = self.tiles[indices_pair[1]]

        with rasterio.open(tile_info_1['path']) as tile_file:
            tile_1 = tile_file.read([1, 2, 3])  # Reading the first three bands

        with rasterio.open(tile_info_2['path']) as tile_file:
            tile_2 = tile_file.read([1, 2, 3])  # Reading the first three bands

        category_id_1 = tile_info_1['labels'][0]['category_id']
        category_id_2 = tile_info_2['labels'][0]['category_id']

        if self.transform:
            transformed_1 = self.transform(image=tile_1.transpose((1, 2, 0)))
            transformed_image_1 = transformed_1['image'].transpose((2, 0, 1))
            transformed_2 = self.transform(image=tile_2.transpose((1, 2, 0)))
            transformed_image_2 = transformed_2['image'].transpose((2, 0, 1))
        else:
            transformed_image_1 = tile_1
            transformed_image_2 = tile_2

        return transformed_image_1, transformed_image_2, category_id_1, category_id_2
