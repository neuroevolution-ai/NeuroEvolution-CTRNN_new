from processing_handlers.i_processing_handler import IProcessingHandler


class SequentialHandler(IProcessingHandler):

    def __init__(self, number_of_workers):
        super().__init__(number_of_workers)

    def map(self, func, *iterable):
        return map(func, *iterable)
