from abc import ABC, abstractmethod

class EmbeddingModel(ABC):

    def __init__(self):
        super(AbstractOperation, self).__init__()

    @abstractmethod
    def build(self):
        pass

    @abstractmethod
    def loss(self):
        pass

    @abstractmethod
    def setup(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def test(self):
        pass
