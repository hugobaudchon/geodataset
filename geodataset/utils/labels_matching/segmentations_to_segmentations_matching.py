from warnings import warn

import geopandas as gpd


def segmentations_to_segmentations_matching(preds_segmentations, truth_segmentations, output_path, iou_threshold):
    # Check if the segmentations and truth_boxes are in the same crs
    if preds_segmentations.crs != truth_segmentations.crs:
        preds_segmentations = preds_segmentations.to_crs(truth_segmentations.crs)

    preds_segmentations['geometry'] = preds_segmentations.buffer(0)

    labeled_segmentations = []

    for i, box in truth_segmentations.iterrows():
        # get segmentations intersecting the box
        intersecting_segmentations = preds_segmentations[preds_segmentations.intersects(box.geometry)].copy(deep=True)

        # get the area of the intersections
        intersecting_segmentations['intersecting_area'] = intersecting_segmentations.intersection(box.geometry).area

        intersecting_segmentations['intersecting_area_ratio'] = intersecting_segmentations['intersecting_area'] / intersecting_segmentations['geometry'].area
        intersecting_segmentations['iou'] = intersecting_segmentations['intersecting_area'] / (intersecting_segmentations['geometry'].area + box.geometry.area - intersecting_segmentations['intersecting_area'])

        segmentations_keep = intersecting_segmentations[intersecting_segmentations['iou'] > iou_threshold].index

        for segmentation_id in segmentations_keep:
            labeled_segmentation = box.copy(deep=True)
            labeled_segmentation.geometry = preds_segmentations.loc[segmentation_id].geometry
            labeled_segmentations.append(labeled_segmentation)

    labeled_segmentations_gdf = gpd.GeoDataFrame(labeled_segmentations, crs=truth_segmentations.crs)

    print("Number of segmentations in truth:", len(truth_segmentations))
    print("Number of predicted segmentations matched to truth segmentations:", len(labeled_segmentations_gdf))

    labeled_segmentations_gdf.to_file(output_path)

    print(f'Labeled segmentations saved to {output_path}.\n')
    warn(f'Please make sure to double check the segments in a GIS software before using them!!!')


if __name__ == '__main__':
    preds_segmentations_path = '/infer/20220929_bci50ha_p4pro_rgb/segmenter_aggregator_output/20220929_bci50ha_p4pro_rgb_gr0p07_infersegmenteraggregator.geojson'
    truth_segmentations_path = '/Data/raw/panama/20220929_bci50ha_p4pro/20220929_bci50ha_p4pro_labels_masks.gpkg'
    iou_threshold = 0.50
    output_path = f'/Data/segmentations_SAM_matched/20220929_bci50ha_p4pro_rgb_SAMthreshold{str(iou_threshold).replace(".", "p")}.gpkg'

    segmentations = gpd.read_file(preds_segmentations_path)
    truth_segmentations = gpd.read_file(truth_segmentations_path)

    segmentations_to_segmentations_matching(segmentations, truth_segmentations, output_path, iou_threshold)
