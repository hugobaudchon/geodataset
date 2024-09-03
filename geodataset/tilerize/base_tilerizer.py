from abc import ABC, abstractmethod

class BaseTilerizer(ABC):
    @abstractmethod
    def tilerize(self, **kwargs):
        pass

    @abstractmethod
    def get_tile(self, **kwargs):
        pass
    
    @abstractmethod
    def _generate_tile_metadata(self, **kwargs):
        pass

class LabeledTilerizer(BaseTilerizer):
    @abstractmethod
    def _load_labels(self, **kwargs):
        pass
    

