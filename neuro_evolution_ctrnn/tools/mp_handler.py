from multiprocessing import Pool


class MPHandler:

    def __init__(self, number_of_workers):
        self.pool = Pool(processes=number_of_workers)

    def map(self, func, *iterable):
        return self.pool.starmap(func, zip(*iterable))

    def cleanup(self):
        self.pool.close()
        self.pool.join()
