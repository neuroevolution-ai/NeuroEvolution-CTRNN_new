from multiprocessing import Pool


class MPHandler:

    def __init__(self, number_of_workers):
        self.number_of_workers = number_of_workers

    def map(self, func, *iterable):
        with Pool(self.number_of_workers) as pool:
            result = pool.starmap(func, zip(*iterable))
        return result

    def cleanup(self):
        # Pool cleans itself up
        pass
