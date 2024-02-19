from abc import abstractmethod, ABC


class BaseDataset(ABC):
    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, index):
        pass
