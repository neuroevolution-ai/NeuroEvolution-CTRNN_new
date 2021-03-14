from concurrent.futures.process import ProcessPoolExecutor

from processing_handlers.i_processing_handler import IProcessingHandler


class ConcurrentHandler(IProcessingHandler):

    def __init__(self, number_of_workers):
        super().__init__(number_of_workers)

    def map(self, func, *iterable):
        return self.pool.map(func, *iterable)

    def init_framework(self):
        self.pool = ProcessPoolExecutor(self.number_of_workers)

    def cleanup_framework(self):
        self.pool.shutdown()
