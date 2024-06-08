from warnings import warn

import geopandas as gpd


def main(segmentations, truth_boxes, output_path, iou_threshold):
    # Check if the segmentations and truth_boxes are in the same crs
    if segmentations.crs != truth_boxes.crs:
        segmentations = segmentations.to_crs(truth_boxes.crs)

    labeled_segmentations = []

    for i, box in truth_boxes.iterrows():
        # get segmentations intersecting the box
        intersecting_segmentations = segmentations[segmentations.intersects(box.geometry)].copy(deep=True)

        # get the area of the intersections
        intersecting_segmentations['intersecting_area'] = intersecting_segmentations.intersection(box.geometry).area

        intersecting_segmentations['intersecting_area_ratio'] = intersecting_segmentations['intersecting_area'] / intersecting_segmentations['geometry'].area
        intersecting_segmentations['iou'] = intersecting_segmentations['intersecting_area'] / (intersecting_segmentations['geometry'].area + box.geometry.area - intersecting_segmentations['intersecting_area'])

        segmentations_keep = intersecting_segmentations[intersecting_segmentations['iou'] > iou_threshold].index

        for segmentation_id in segmentations_keep:
            labeled_segmentation = box.copy(deep=True)
            labeled_segmentation.geometry = segmentations.loc[segmentation_id].geometry
            labeled_segmentations.append(labeled_segmentation)

    labeled_segmentations_gdf = gpd.GeoDataFrame(labeled_segmentations, crs=truth_boxes.crs)

    labeled_segmentations_gdf.to_file(output_path)

    print(f'Labeled segmentations saved to {output_path}.\n')
    warn(f'Please make sure to double check the segments in a GIS software before using them!!!')


if __name__ == '__main__':
    segmentations_path = '/infer/20240130_zf2quad_m3m_rgb_BOTH_SCORES_MULTIPLIED/segmenter_aggregator_output/20240130_zf2quad_m3m_rgb_gr0p07_infersegmenteraggregator.geojson'
    truth_boxes_path = '/Data/raw/brazil_zf2/20240130_zf2quad_m3m_rgb_labels_boxes_species.gpkg'
    output_path = '/Data/raw/labeled_segmentations/20240130_zf2quad_m3m_rgb_labels_segmentationsSAM_species.gpkg'
    iou_threshold = 0.20

    segmentations = gpd.read_file(segmentations_path)
    truth_boxes = gpd.read_file(truth_boxes_path)

    main(segmentations, truth_boxes, output_path, iou_threshold)