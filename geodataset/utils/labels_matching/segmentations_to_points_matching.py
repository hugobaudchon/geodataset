from pathlib import Path
from warnings import warn

import cv2
import geopandas as gpd
import numpy as np
from shapely import MultiPolygon, Polygon, box
from shapely.affinity import rotate
import largestinteriorrectangle as lir


def normalize_polygon(polygon, target_range=(0, 100)):
    minx, miny, maxx, maxy = polygon.bounds

    # Calculate the scaling factor
    scale_x = (target_range[1] - target_range[0]) / (maxx - minx)
    scale_y = (target_range[1] - target_range[0]) / (maxy - miny)

    # Use the smaller scaling factor to maintain aspect ratio
    scale = min(scale_x, scale_y)

    # Calculate the translation values
    trans_x = -minx * scale + target_range[0]
    trans_y = -miny * scale + target_range[0]

    def scale_and_translate(x, y, scale, trans_x, trans_y):
        return x * scale + trans_x, y * scale + trans_y

    # Apply the transformation to all coordinates
    normalized_coords = [scale_and_translate(x, y, scale, trans_x, trans_y) for x, y in polygon.exterior.coords]

    # Create a new polygon with the normalized coordinates
    normalized_polygon = Polygon(normalized_coords)

    return normalized_polygon, scale, trans_x, trans_y


def revert_normalization(normalized_polygon, scale, trans_x, trans_y):
    def unscale_and_untranslate(x, y, scale, trans_x, trans_y):
        return (x - trans_x) / scale, (y - trans_y) / scale

    # Apply the inverse transformation to all coordinates
    original_coords = [unscale_and_untranslate(x, y, scale, trans_x, trans_y) for x, y in normalized_polygon.exterior.coords]

    # Create a new polygon with the original coordinates
    original_polygon = Polygon(original_coords)

    return original_polygon


def largest_inner_rectangle(geometry):
    # Convert MultiPolygon to Polygon by taking the largest part
    if isinstance(geometry, MultiPolygon):
        geometry = max(geometry.geoms, key=lambda a: a.area)

    # Normalize the polygon to a target range
    normalized_polygon, scale, trans_x, trans_y = normalize_polygon(geometry)

    # Polygon to boolean mask
    mask = np.zeros((100, 100), dtype=np.uint8)
    exterior_coords = np.array([list(normalized_polygon.exterior.coords)]).astype(np.int32)
    cv2.fillPoly(mask, exterior_coords, 1)
    mask = mask.astype(bool)

    rectangle = lir.lir(mask)

    rectangle_shapely = box(rectangle[0], rectangle[1], rectangle[0]+rectangle[2], rectangle[1]+rectangle[3])

    # Revert transform applied in normalize_polygon
    best_rectangle = revert_normalization(rectangle_shapely, scale, trans_x, trans_y)

    return best_rectangle


def segmentations_to_points_matching(segmentations, truth_points, output_path, relative_dist_to_centroid_threshold, use_largest_inner_rectangle):
    if segmentations.crs != truth_points.crs:
        segmentations = segmentations.to_crs(truth_points.crs)

    # only keep segmentations intersecting the truth points
    segmentations['geometry'] = segmentations.buffer(0)
    segmentations = segmentations[segmentations.intersects(truth_points.unary_union)]

    if use_largest_inner_rectangle:
        # get the largest rectangle that fits inside the segmentation
        print("Calculating largest inner rectangles...")
        segmentations['largest_inner_rectangle'] = segmentations['geometry'].astype(object).apply(largest_inner_rectangle)
        trees_polygons = segmentations.copy(deep=True)
        trees_polygons['geometry'] = trees_polygons['largest_inner_rectangle']
        print("Done.\n")
    else:
        trees_polygons = segmentations.copy(deep=True)

    # Only keep the largest geom if multipolygon
    trees_polygons['geometry'] = trees_polygons['geometry'].apply(lambda x: x if x.geom_type == 'Polygon' else (x if x.area == max([geom.area for geom in x.geoms]) else max(x.geoms, key=lambda a: a.area)))

    # get the max distance between 2 points of the segmentation
    trees_polygons['max_point_distance'] = trees_polygons['geometry'].apply(lambda x: x.exterior.distance(x.centroid))

    labeled_segmentations = []

    for i, segmentation in trees_polygons.iterrows():
        intersecting_points = truth_points[truth_points.intersects(segmentation.geometry)].copy(deep=True)

        if intersecting_points.empty:
            continue

        intersecting_points['dist_to_centroid'] = intersecting_points.centroid.distance(segmentation.geometry.centroid)
        intersecting_points['relative_dist_to_centroid'] = intersecting_points['dist_to_centroid'] / segmentation['max_point_distance']

        best_point_idx = intersecting_points['relative_dist_to_centroid'].idxmin()

        if intersecting_points.loc[best_point_idx, 'relative_dist_to_centroid'] < relative_dist_to_centroid_threshold:
            labeled_segmentation = truth_points.loc[best_point_idx].copy(deep=True)
            labeled_segmentation['geometry'] = segmentations.loc[i].geometry
            labeled_segmentations.append(labeled_segmentation)

    if len(labeled_segmentations) == 0:
        raise Exception("No segmentations matched the truth points. Please check the relative distance to the centroid.")

    labeled_segmentations_gdf = gpd.GeoDataFrame(labeled_segmentations, crs=truth_points.crs)

    print("Number of points in truth:", len(truth_points))
    print("Number of predicted segmentations matched to truth segmentations:", len(labeled_segmentations_gdf))

    labeled_segmentations_gdf.to_file(output_path)

    print(f'Labeled segmentations saved to {output_path}.\n')
    warn(f'Please make sure to double check the segments in a GIS software before using them!!!')


if __name__ == '__main__':
    segmentations_paths = [
        '/media/hugobaudchon/4 TB/XPrize/Data/SAM_segmentations_from_boxes/brazil_zf2/20240131_zf2campirana_m3m_rgb_gr0p08_infersegmenteraggregator.gpkg',
        '/media/hugobaudchon/4 TB/XPrize/Data/SAM_segmentations_from_boxes/equator/20170810_transectotoni_mavicpro_rgb_gr0p08_infersegmenteraggregator.gpkg',
        '/media/hugobaudchon/4 TB/XPrize/Data/SAM_segmentations_from_boxes/equator/20230525_tbslake_m3e_rgb_gr0p08_infersegmenteraggregator.gpkg',
        '/media/hugobaudchon/4 TB/XPrize/Data/SAM_segmentations_from_boxes/equator/20231018_inundated_m3e_rgb_gr0p08_infersegmenteraggregator.gpkg',
        '/media/hugobaudchon/4 TB/XPrize/Data/SAM_segmentations_from_boxes/equator/20231018_pantano_m3e_rgb_gr0p08_infersegmenteraggregator.gpkg',
        '/media/hugobaudchon/4 TB/XPrize/Data/SAM_segmentations_from_boxes/equator/20231018_terrafirme_m3e_rgb_gr0p08_infersegmenteraggregator.gpkg',
    ]
    truth_points_paths = [
        '/media/hugobaudchon/4 TB/XPrize/Data/raw/brazil_zf2/labels_species/20240131_zf2campinarana_labels_points_species.gpkg',
        '/media/hugobaudchon/4 TB/XPrize/Data/raw/equator/labels_points_species/20170810_transectotoni_mavicpro_labels_points.gpkg',
        '/media/hugobaudchon/4 TB/XPrize/Data/raw/equator/labels_points_species/20230525_tbslake_m3e_labels_points.gpkg',
        '/media/hugobaudchon/4 TB/XPrize/Data/raw/equator/labels_points_species/20231018_inundated_m3e_labels_points.gpkg',
        '/media/hugobaudchon/4 TB/XPrize/Data/raw/equator/labels_points_species/20231018_pantano_m3e_labels_points.gpkg',
        '/media/hugobaudchon/4 TB/XPrize/Data/raw/equator/labels_points_species/20231018_terrafirme_m3e_labels_points.gpkg',
    ]
    relative_dist_to_centroid_threshold = 1.0

    output_folder = '/media/hugobaudchon/4 TB/XPrize/Data/SAM_segmentations_matched'

    use_largest_inner_rectangle = True

    for segmentations_path, truth_points_path in zip(segmentations_paths, truth_points_paths):
        output_path = f'{output_folder}/{Path(segmentations_path).stem}_SAMthreshold{str(relative_dist_to_centroid_threshold).replace(".", "p")}.gpkg'
        segmentations = gpd.read_file(segmentations_path)
        truth_points = gpd.read_file(truth_points_path)

        segmentations_to_points_matching(segmentations, truth_points, output_path, relative_dist_to_centroid_threshold, use_largest_inner_rectangle)
