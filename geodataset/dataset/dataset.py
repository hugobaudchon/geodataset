import json
import os
from pathlib import Path
from typing import List

from geodataset.geodata.base_geodata import BaseGeoData
from geodataset.labels.raster_labels import RasterDetectionLabels


class RasterDataset:
        a = 0


class LabeledRasterDataset(RasterDataset):
    def __init__(self, data_folders_paths: List[Path]):
        self.data_folders_paths = data_folders_paths

    def load_and_combine_coco_datasets(folders):
        combined_annotations = {
            "images": [],
            "annotations": [],
            "categories": None  # Assuming categories are consistent across datasets
        }

        next_image_id = 1
        next_annotation_id = 1

        for folder in folders:
            annotation_file = os.path.join(folder, "annotations.json")

            with open(annotation_file, 'r') as f:
                data = json.load(f)

                # Initialize categories if not already set
                if combined_annotations["categories"] is None:
                    combined_annotations["categories"] = data["categories"]

                for image in data["images"]:
                    old_image_id = image["id"]
                    image["id"] = next_image_id
                    next_image_id += 1

                    # Copy image file to new destination if necessary
                    # src_image_path = os.path.join(folder, "tiles", image["file_name"])
                    # dest_image_path = os.path.join("combined_dataset/tiles", image["file_name"])
                    # copyfile(src_image_path, dest_image_path)

                    combined_annotations["images"].append(image)

                    # Update annotations with new image_id
                    for annotation in data["annotations"]:
                        if annotation["image_id"] == old_image_id:
                            annotation["image_id"] = image["id"]
                            annotation["id"] = next_annotation_id
                            next_annotation_id += 1
                            combined_annotations["annotations"].append(annotation)

        # Save combined annotations to a new file
        with open('combined_annotations.json', 'w') as outfile:
            json.dump(combined_annotations, outfile)

        return combined_annotations
