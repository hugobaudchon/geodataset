from warnings import warn

import geopandas as gpd
from shapely.validation import explain_validity


def segmentations_to_boxes_matching(segmentations, truth_boxes, output_path, iou_threshold):
    # Check if the segmentations and truth_boxes are in the same crs
    if segmentations.crs != truth_boxes.crs:
        segmentations = segmentations.to_crs(truth_boxes.crs)

    segmentations.geometry = segmentations.buffer(0)
    truth_boxes.geometry = truth_boxes.buffer(0)

    labeled_segmentations = []

    for i, box in truth_boxes.iterrows():
        # get segmentations intersecting the box
        intersecting_segmentations = segmentations[segmentations.intersects(box.geometry)].copy(deep=True)

        # get the area of the intersections
        intersecting_segmentations['box_area'] = box.geometry.area
        intersecting_segmentations['segmentation_area'] = segmentations.loc[intersecting_segmentations.index, 'geometry'].area
        intersecting_segmentations['intersecting_area'] = intersecting_segmentations.intersection(box.geometry).area

        intersecting_segmentations['union_area'] = segmentations.loc[intersecting_segmentations.index, 'geometry'].area + box.geometry.area - intersecting_segmentations['intersecting_area']
        intersecting_segmentations['iou'] = intersecting_segmentations['intersecting_area'] / intersecting_segmentations['union_area']

        segmentations_keep = intersecting_segmentations[intersecting_segmentations['iou'] > iou_threshold].index

        for segmentation_id in segmentations_keep:
            labeled_segmentation = box.copy(deep=True)
            labeled_segmentation.geometry = segmentations.loc[segmentation_id].geometry
            labeled_segmentations.append(labeled_segmentation)

    labeled_segmentations_gdf = gpd.GeoDataFrame(labeled_segmentations, crs=truth_boxes.crs)

    print("Number of points in truth:", len(truth_boxes))
    print("Number of predicted segmentations matched to truth segmentations:", len(labeled_segmentations_gdf))

    labeled_segmentations_gdf.to_file(output_path)

    print(f'Labeled segmentations saved to {output_path}.\n')
    warn(f'Please make sure to double check the segments in a GIS software before using them!!!')


if __name__ == '__main__':
    segmentations_path = '/media/hugobaudchon/4 TB/XPrize/Data/SAM_segmentations_from_boxes/brazil_zf2/20240130_zf2quad_m3m_rgb_gr0p08_infersegmenteraggregator.gpkg'
    truth_boxes_path = '/media/hugobaudchon/4 TB/XPrize/Data/raw/brazil_zf2/labels_species/20240130_zf2quad_m3m_rgb_labels_boxes_species.gpkg'
    iou_threshold = 0.20
    output_path = f'/media/hugobaudchon/4 TB/XPrize/Data/SAM_segmentations_matched/20240130_zf2quad_m3m_rgb_SAMthreshold{str(iou_threshold).replace(".", "p")}_species.gpkg'

    segmentations = gpd.read_file(segmentations_path)
    truth_boxes = gpd.read_file(truth_boxes_path)

    segmentations_to_boxes_matching(segmentations, truth_boxes, output_path, iou_threshold)
