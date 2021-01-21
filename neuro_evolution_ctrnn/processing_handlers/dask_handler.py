import logging
from typing import Optional, Union

from dask.distributed import Client, Worker, WorkerPlugin, LocalCluster

from brains.continuous_time_rnn import ContinuousTimeRNN
from processing_handlers.i_processing_handler import IProcessingHandler


class _CreatorPlugin(WorkerPlugin):
    """Initiated global states for every worker."""

    def __init__(self, brain_class: Union[ContinuousTimeRNN]):
        # Get Class-Variable from main thread to re-initialize them on worker
        self.brain_class = brain_class
        self.brain_class_state = brain_class.get_class_state()

    def setup(self, worker):
        logging.debug("setting up classes for worker: " + str(worker))
        self.brain_class.set_class_state(**self.brain_class_state)

    def teardown(self, worker: Worker):
        pass

    def transition(self, key: str, start: str, finish: str, **kwargs):
        # called whenever worker gets new task, i think
        pass


class DaskHandler(IProcessingHandler):
    """This class wraps all Dask related functions."""

    def __init__(self, number_of_workers, brain_class, worker_log_level=logging.WARNING):
        super().__init__(number_of_workers)
        self._client: Optional[Client] = None
        self._cluster: Optional[LocalCluster] = None

        self.brain_class = brain_class
        self.worker_log_level = worker_log_level

    def init_framework(self):
        if self._client:
            raise RuntimeError("Dask client already initialized.")

        # threads_per_worker must be one, because atari-env is not thread-safe.
        # And because lower the thread-count from the default, we must increase the number of workers
        self._cluster = LocalCluster(processes=True, asynchronous=False, threads_per_worker=1,
                                     silence_logs=self.worker_log_level,
                                     n_workers=self.number_of_workers,
                                     memory_pause_fraction=False,
                                     lifetime='1 hour', lifetime_stagger='5 minutes', lifetime_restart=True,
                                     interface="lo")
        self._client = Client(self._cluster)
        self._client.register_worker_plugin(_CreatorPlugin(self.brain_class), name="creator-plugin")
        logging.info("Dask dashboard available at port: " + str(self._client.scheduler_info()["services"]["dashboard"]))

    def map(self, func, *iterable):
        if not self._client:
            raise RuntimeError("Dask client not initialized. Call \"init_framework\" before calling \"map\"")
        return self._client.gather(self._client.map(func, *iterable))

    def cleanup_framework(self):
        self._client.shutdown()
