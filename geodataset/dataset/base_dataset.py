from abc import ABC, abstractmethod


class BaseDataset(ABC):
    @abstractmethod
    def __getitem__(self, idx: int):
        pass

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __iter__(self):
        pass
