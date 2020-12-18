from processing_handlers.i_processing_handler import IProcessingHandler
from ray.util.multiprocessing import Pool
import ray  # TODO add ray to requirements


class RayHandler(IProcessingHandler):

    def __init__(self, number_of_workers, class_cb, brain_class):
        super().__init__(number_of_workers)
        self.pool = None

    def init_framework(self):
        super().init_framework()
        ray.init(local_mode=False)
        self.pool = Pool(processes=self.number_of_workers)

    def map(self, func, *iterable):
        result = self.pool.starmap(func, list(zip(*iterable)))
        return result

    def cleanup_framework(self):
        super().cleanup_framework()
        self.pool.close()
        self.pool.join()
