from abc import ABCMeta, abstractmethod

class EmbeddingModel(metaclass=ABCMeta):

    @abstractmethod
    def build(self):
        pass

    @abstractmethod
    def _loss(self):
        pass

    @abstractmethod
    def train(self):
        pass

class Classifier(metaclass=ABCMeta):

    @abstractmethod
    def build(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def predict(self):
        pass
