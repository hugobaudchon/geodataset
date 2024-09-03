from abc import ABC, abstractmethod

class BaseTilerizer(ABC):
    @abstractmethod
    def tilerize(self):
        pass

class LabeledTilerizer(BaseTilerizer):
    @abstractmethod
    def tilerize(self):
        pass