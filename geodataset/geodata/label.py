from typing import List

import shapely
from pyproj import Transformer
from shapely import Polygon, box
from shapely.ops import transform
import rasterio
from functools import partial
import pyproj


class PolygonLabel:
    def __init__(self,
                 polygon: shapely.Polygon,
                 category: str or int or None,
                 crs: pyproj.CRS = None):
        self.polygon = polygon
        self.category = category

        self.crs = crs
        self.affine_transform = None
        self.scale_factor = 1

    @classmethod
    def from_bbox(cls,
                  bbox: List[int],
                  category: str or int = None):

        assert bbox[0] <= bbox[2] and bbox[1] <= bbox[3], \
            "The polygon bbox doesn't seem to be in the format [xmin, ymin, xmax, ymax]."

        polygon = Polygon([(bbox[0], bbox[1]),
                           (bbox[0], bbox[3]),
                           (bbox[2], bbox[3]),
                           (bbox[2], bbox[1])])

        return cls(polygon=polygon, category=category)

    def apply_crs(self, target_crs: pyproj.CRS):
        assert self.crs is not None, ("To apply a new CRS the polygon must already have a CRS. "
                                      "Maybe you forgot to specify the CRS when instantiating the PolygonLabel?")

        if self.crs == target_crs:
            return

        # Create a transformer to convert from the original CRS to the target CRS
        transformer = Transformer.from_crs(self.crs, target_crs, always_xy=True)
        transformed_polygon_coords = [transformer.transform(x, y) for x, y in self.polygon.exterior.coords]
        self.polygon = Polygon(transformed_polygon_coords)

    def apply_crs_to_pixel_transform(self, associated_raster_transform: rasterio.transform.Affine):
        assert self.crs is not None, "To apply a CRS to pixel transform, it is expected that there is a CRS."
        assert self.affine_transform is None, ("Currently chaining transform operations is not supported."
                                               " Only one transform can be applied to a polygon label.")

        # Use the inverse of the associated raster's affine transform
        inv_transform = ~associated_raster_transform

        # Function to apply the inverse transform to each coordinate
        def transform_coord(x, y, transform_fct):
            # Applying the inverse transform to the coordinate
            x, y = transform_fct * (x, y)
            return x, y

        # Apply the transformation to each point in the polygon
        transformed_polygon = transform(partial(transform_coord, transform_fct=inv_transform), self.polygon)

        # Update the polygon with the transformed coordinates
        self.polygon = transformed_polygon
        self.affine_transform = associated_raster_transform

    def apply_scale_factor(self, scale_factor: float):
        if scale_factor == 1.0:
            return
        scaled_coords = [(x * scale_factor, y * scale_factor) for x, y in self.polygon.exterior.coords]
        self.polygon = Polygon(scaled_coords)
        self.scale_factor *= scale_factor

    def get_bbox(self):
        return self.polygon.bounds

    @staticmethod
    def window_to_bbox(window: rasterio.windows.Window):
        """
        Convert a rasterio window to a bounding box [minx, miny, maxx, maxy] in pixel coordinates.
        """
        minx, miny = window.col_off, window.row_off
        maxx, maxy = window.col_off + window.width, window.row_off + window.height
        return [minx, miny, maxx, maxy]

    @staticmethod
    def calculate_area(bounds):
        """Calculate the area of a rectangle defined by bounds [minx, miny, maxx, maxy]."""
        minx, miny, maxx, maxy = bounds
        return (maxx - minx) * (maxy - miny)

    def is_in_window(self,
                     window: rasterio.windows.Window,
                     min_intersection_ratio: float):
        _, intersection_area = self._get_intersection_with_window(window)
        intersection_ratio = intersection_area / self.polygon.area

        if intersection_ratio > min_intersection_ratio:
            return True
        else:
            return False

    def is_bbox_in_window(self, window, min_intersection_ratio):
        # Faster version of is_in_window which only works on the bbox instead of polygon itself
        # (useful for detection tasks)

        crop_region = self.window_to_bbox(window)
        bbox = self.get_bbox()

        # Calculate intersection
        intersect_minx = max(bbox[0], crop_region[0])
        intersect_miny = max(bbox[1], crop_region[1])
        intersect_maxx = min(bbox[2], crop_region[2])
        intersect_maxy = min(bbox[3], crop_region[3])

        # Check if there is an intersection
        if intersect_minx < intersect_maxx and intersect_miny < intersect_maxy:
            intersect_area = self.calculate_area([intersect_minx, intersect_miny, intersect_maxx, intersect_maxy])
            bbox_area = self.calculate_area(bbox)
            ratio = intersect_area / bbox_area

            return ratio >= min_intersection_ratio
        else:
            return False

    def get_cropped_polygon_label(self, window: rasterio.windows.Window):
        intersection_polygon, _ = self._get_intersection_with_window(window)

        cropped_polygon_label = PolygonLabel(
            polygon=intersection_polygon,
            category=self.category,
            crs=self.crs)
        cropped_polygon_label.affine_transform = self.affine_transform
        cropped_polygon_label.scale_factor = self.scale_factor

        return cropped_polygon_label

    def get_cropped_bbox_polygon_label(self, window):
        # Faster version of get_cropped_polygon_label which only works on the bbox instead of polygon itself
        # (useful for detection tasks)

        crop_region = self.window_to_bbox(window)
        bbox = self.get_bbox()

        # Calculate intersection
        intersect_minx = max(bbox[0], crop_region[0])
        intersect_miny = max(bbox[1], crop_region[1])
        intersect_maxx = min(bbox[2], crop_region[2])
        intersect_maxy = min(bbox[3], crop_region[3])

        # Check if there is an intersection and it's meaningful
        if intersect_minx < intersect_maxx and intersect_miny < intersect_maxy:
            # Adjust coordinates to be relative to the window
            rel_intersect_bbox = [
                intersect_minx - crop_region[0],
                intersect_miny - crop_region[1],
                intersect_maxx - crop_region[0],
                intersect_maxy - crop_region[1]
            ]

            # Create and return a new PolygonLabel with the cropped bbox
            cropped_polygon = Polygon([
                (rel_intersect_bbox[0], rel_intersect_bbox[1]),
                (rel_intersect_bbox[0], rel_intersect_bbox[3]),
                (rel_intersect_bbox[2], rel_intersect_bbox[3]),
                (rel_intersect_bbox[2], rel_intersect_bbox[1])
            ])
            return PolygonLabel(polygon=cropped_polygon, category=self.category, crs=self.crs)
        else:
            # No meaningful intersection; could return None or handle differently based on your requirements
            return None

    def _get_intersection_with_window(self, window: rasterio.windows.Window):
        win_bounds = self.window_to_bbox(window=window)
        window_polygon = box(*win_bounds)

        intersection_polygon = window_polygon.intersection(self.polygon)
        intersection_area = intersection_polygon.area
        return intersection_polygon, intersection_area

    def to_coco(self, image_id: int, category_id_map: dict):
        """
        Convert the polygon label to a COCO-format dictionary.

        Args:
            image_id (int): The ID of the image this polygon is associated with.
            category_id_map (dict): A mapping from category names or IDs used in this class to the corresponding COCO category IDs.

        Returns:
            dict: A dictionary formatted according to COCO specifications.
        """

        # Ensure the category ID is mapped correctly for COCO
        if self.category in category_id_map:
            category_id = category_id_map[self.category]
        else:
            raise ValueError(f"Category '{self.category}' not found in category ID map.")

        # Convert the polygon's exterior coordinates to the format expected by COCO.
        # COCO expects a flat list of coordinates for segmentation: [x1, y1, x2, y2, ..., xn, yn]
        segmentation = [coord for xy in self.polygon.exterior.coords for coord in xy]

        # Calculate the area of the polygon
        area = self.polygon.area

        # Get the bounding box in COCO format: [x, y, width, height]
        bbox = list(self.get_bbox())
        bbox_coco_format = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]

        # Construct the COCO representation
        coco_annotation = {
            "segmentation": [segmentation],  # COCO expects a list of polygons, each a list of coordinates
            "area": area,
            "iscrowd": 0,  # Assuming this polygon represents a single object (not a crowd)
            "image_id": image_id,
            "bbox": bbox_coco_format,
            "category_id": category_id,
            "id": None  # ID should be assigned externally if tracking specific annotations
        }

        return coco_annotation


