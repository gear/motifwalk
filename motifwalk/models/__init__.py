from abc import ABCMeta, abstractmethod

class EmbeddingModel(metaclass=ABCMeta):

    @abstractmethod
    def build(self):
        pass

    @abstractmethod
    def loss(self):
        pass

    @abstractmethod
    def train(self):
        pass
