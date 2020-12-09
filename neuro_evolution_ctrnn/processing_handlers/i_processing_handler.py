import abc


class IProcessingHandler(abc.ABC):

    def __init__(self, number_of_workers):
        self.number_of_workers = number_of_workers

    def init_framework(self):
        pass

    @abc.abstractmethod
    def map(self, func, *iterable):
        pass

    def cleanup_framework(self):
        pass
