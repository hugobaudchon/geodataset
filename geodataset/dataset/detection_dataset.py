from geodataset.geodata.geodata import BaseGeoData
from geodataset.labels.detection_labels import DetectionLabels


class LabeledDetectionRasterDataset:
    def __init__(self, data: BaseGeoData, labels: DetectionLabels):
        self.data = data
        self.labels = labels
