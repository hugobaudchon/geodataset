from abc import ABC, abstractmethod

class BaseTilerizer(ABC):
    @abstractmethod
    def tilerize(self):
        pass
    @abstractmethod
    def create_folder(self):
        pass
    @abstractmethod
    def lazy_tilerize(self):
        pass
    @abstractmethod
    def downsample(self):
        pass
    # TODO: Implement this method
    # @abstractmethod
    # def generate_tiles_metadata(self):
    #     pass

class LabeledTilerizer(BaseTilerizer):    
    @abstractmethod
    def load_labels(self):
        pass