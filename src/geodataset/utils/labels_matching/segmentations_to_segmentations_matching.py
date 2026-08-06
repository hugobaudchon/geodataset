from pathlib import Path
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


def file_match():
    preds_segmentations_paths = [
        # '/media/hugobaudchon/4 TB/XPrize/Data/SAM_segmentations_from_boxes/brazil_zf2/20240130_zf2quad_m3m_rgb_gr0p08_infersegmenteraggregator.gpkg',
        # '/media/hugobaudchon/4 TB/XPrize/Data/SAM_segmentations_from_boxes/brazil_zf2/20240131_zf2campirana_m3m_rgb_gr0p08_infersegmenteraggregator.gpkg',
        # '/media/hugobaudchon/4 TB/XPrize/Data/SAM_segmentations_from_boxes/equator/20170810_transectotoni_mavicpro_rgb_gr0p08_infersegmenteraggregator.gpkg',
        # '/media/hugobaudchon/4 TB/XPrize/Data/SAM_segmentations_from_boxes/equator/20230525_tbslake_m3e_rgb_gr0p08_infersegmenteraggregator.gpkg',
        # '/media/hugobaudchon/4 TB/XPrize/Data/SAM_segmentations_from_boxes/equator/20231018_inundated_m3e_rgb_gr0p08_infersegmenteraggregator.gpkg',
        # '/media/hugobaudchon/4 TB/XPrize/Data/SAM_segmentations_from_boxes/equator/20231018_pantano_m3e_rgb_gr0p08_infersegmenteraggregator.gpkg',
        # '/media/hugobaudchon/4 TB/XPrize/Data/SAM_segmentations_from_boxes/equator/20231018_terrafirme_m3e_rgb_gr0p08_infersegmenteraggregator.gpkg',
        '/media/hugobaudchon/4 TB/XPrize/Data/SAM_segmentations_from_boxes/quebec_trees/2021_09_02_sbl_z1_rgb_cog_gr0p08_infersegmenteraggregator.gpkg',
        '/media/hugobaudchon/4 TB/XPrize/Data/SAM_segmentations_from_boxes/quebec_trees/2021_09_02_sbl_z2_rgb_cog_gr0p08_infersegmenteraggregator.gpkg',
        '/media/hugobaudchon/4 TB/XPrize/Data/SAM_segmentations_from_boxes/quebec_trees/2021_09_02_sbl_z3_rgb_cog_gr0p08_infersegmenteraggregator.gpkg',
        '/media/hugobaudchon/4 TB/XPrize/Data/SAM_segmentations_from_boxes/panama/20200801_bci50ha_p4pro_rgb_gr0p08_infersegmenteraggregator.gpkg',
        '/media/hugobaudchon/4 TB/XPrize/Data/SAM_segmentations_from_boxes/panama/20220929_bci50ha_p4pro_rgb_gr0p08_infersegmenteraggregator.gpkg'
    ]
    truth_segmentations_paths = [
        # '/media/hugobaudchon/4 TB/XPrize/Data/raw/brazil_zf2/labels_species/20240130_zf2quad_m3m_rgb_labels_boxes_species.gpkg',
        # '/media/hugobaudchon/4 TB/XPrize/Data/raw/brazil_zf2/labels_species/20240131_zf2campinarana_labels_points_species.gpkg',
        # '/media/hugobaudchon/4 TB/XPrize/Data/raw/equator/labels_points_species/20170810_transectotoni_mavicpro_labels_points.gpkg',
        # '/media/hugobaudchon/4 TB/XPrize/Data/raw/equator/labels_points_species/20230525_tbslake_m3e_labels_points.gpkg',
        # '/media/hugobaudchon/4 TB/XPrize/Data/raw/equator/labels_points_species/20231018_inundated_m3e_labels_points.gpkg',
        # '/media/hugobaudchon/4 TB/XPrize/Data/raw/equator/labels_points_species/20231018_pantano_m3e_labels_points.gpkg',
        # '/media/hugobaudchon/4 TB/XPrize/Data/raw/equator/labels_points_species/20231018_terrafirme_m3e_labels_points.gpkg',
        '/media/hugobaudchon/4 TB/XPrize/Data/raw/quebec_trees_dataset/quebec_trees_dataset_2021-09-02/Z1_polygons.gpkg',
        '/media/hugobaudchon/4 TB/XPrize/Data/raw/quebec_trees_dataset/quebec_trees_dataset_2021-09-02/Z2_polygons.gpkg',
        '/media/hugobaudchon/4 TB/XPrize/Data/raw/quebec_trees_dataset/quebec_trees_dataset_2021-09-02/Z3_polygons.gpkg',
        '/media/hugobaudchon/4 TB/XPrize/Data/raw/panama/20200801_bci50ha_p4pro/20200801_bci50ha_p4pro_labels_masks.gpkg',
        '/media/hugobaudchon/4 TB/XPrize/Data/raw/panama/20220929_bci50ha_p4pro/20220929_bci50ha_p4pro_labels_masks.gpkg'
    ]
    iou_threshold = 0.50

    output_folder = '/media/hugobaudchon/4 TB/XPrize/Data/SAM_segmentations_matched'

    for preds_segmentations_path, truth_segmentations_path in zip(preds_segmentations_paths, truth_segmentations_paths):
        segmentations = gpd.read_file(preds_segmentations_path)
        truth_segmentations = gpd.read_file(truth_segmentations_path)

        output_path = f"{output_folder}/{Path(preds_segmentations_path).stem}_SAMthreshold{str(iou_threshold).replace('.', 'p')}.gpkg"

        segmentations_to_segmentations_matching(segmentations, truth_segmentations, output_path, iou_threshold)


def quebec_trees_match():
    iou_threshold = 0.50
    preds_segmentations_paths = Path('/media/hugobaudchon/4 TB/XPrize/Data/SAM_segmentations/quebec_trees').rglob('*.gpkg')
    z1_truth = Path('/media/hugobaudchon/4 TB/XPrize/Data/raw/quebec_trees_dataset/quebec_trees_dataset_2021-09-02/Z1_polygons.gpkg')
    z2_truth = Path('/media/hugobaudchon/4 TB/XPrize/Data/raw/quebec_trees_dataset/quebec_trees_dataset_2021-09-02/Z2_polygons.gpkg')
    z3_truth = Path('/media/hugobaudchon/4 TB/XPrize/Data/raw/quebec_trees_dataset/quebec_trees_dataset_2021-09-02/Z3_polygons.gpkg')

    for preds_segmentations_path in preds_segmentations_paths:
        if 'z1' in preds_segmentations_path.name:
            truth_segmentations_path = z1_truth
        elif 'z2' in preds_segmentations_path.name:
            truth_segmentations_path = z2_truth
        elif 'z3' in preds_segmentations_path.name:
            truth_segmentations_path = z3_truth
        else:
            raise Exception(f"Unknown truth segmentations for {preds_segmentations_path.name}")

        output_path = f'/media/hugobaudchon/4 TB/XPrize/Data/SAM_segmentations_matched/quebec_trees/{preds_segmentations_path.stem}_SAMthreshold{str(iou_threshold).replace(".", "p")}.gpkg'

        segmentations = gpd.read_file(preds_segmentations_path)
        truth_segmentations = gpd.read_file(truth_segmentations_path)

        segmentations_to_segmentations_matching(segmentations, truth_segmentations, output_path, iou_threshold)


def panama_bci_match():
    iou_threshold = 0.50
    preds_segmentations_paths = list(Path('/media/hugobaudchon/4 TB/XPrize/Data/SAM_segmentations/panama_no_gray').rglob('*.gpkg'))
    aligned_truths = list(Path('/media/hugobaudchon/4 TB/XPrize/Data/panama_BCI_aligned_labels').rglob('*.gpkg'))

    for preds_segmentations_path in preds_segmentations_paths:
        name = "_".join(preds_segmentations_path.name.split('_')[:5])
        print(name)
        truth_segmentations_path = [truth for truth in aligned_truths if name in truth.name][0]

        output_path = f'/media/hugobaudchon/4 TB/XPrize/Data/SAM_segmentations_matched/panama/{preds_segmentations_path.stem}_SAMthreshold{str(iou_threshold).replace(".", "p")}.gpkg'

        segmentations = gpd.read_file(preds_segmentations_path)
        truth_segmentations = gpd.read_file(truth_segmentations_path)

        segmentations_to_segmentations_matching(segmentations, truth_segmentations, output_path, iou_threshold)


if __name__ == '__main__':
    file_match()
    # quebec_trees_match()
    # panama_bci_match()
