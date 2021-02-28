from multiprocessing import Pool

from processing_handlers.i_processing_handler import IProcessingHandler


class MPHandler(IProcessingHandler):

    def __init__(self, number_of_workers):
        super().__init__(number_of_workers)

    def map(self, func, *iterable):
        return self.pool.starmap(func, zip(*iterable))

    def init_framework(self):
        self.pool = Pool(self.number_of_workers)
