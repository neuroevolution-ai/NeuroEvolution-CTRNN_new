from typing import Optional
import logging
from typing import Callable, Union

from dask.distributed import Client, Worker, WorkerPlugin, LocalCluster
from dask.distributed import get_worker

from brains.continuous_time_rnn import ContinuousTimeRNN


class _CreatorPlugin(WorkerPlugin):
    """Initiated global states for every worker."""

    def __init__(self, class_creator_callback: Callable, brain_class: Union[ContinuousTimeRNN]):
        self.callback = class_creator_callback

        # Get Class-Variable from main thread to re-initialize them on worker
        self.brain_class = brain_class
        self.brain_class_state = brain_class.get_class_state()

    def setup(self, worker):
        logging.info("setting up classes for worker: " + str(worker))
        self.callback()
        self.brain_class.set_class_state(**self.brain_class_state)

    def teardown(self, worker: Worker):
        pass

    def transition(self, key: str, start: str, finish: str, **kwargs):
        # called whenever worker gets new task, i think
        pass


class DaskHandler:
    """this class wraps all Dask-Related functions."""
    _client: Optional[Client] = None
    _cluster: Optional[LocalCluster] = None

    @classmethod
    def init_dask(cls, class_cb: Callable, brain_class, n_workers, worker_log_level=logging.WARN):
        if cls._client:
            raise RuntimeError("dask client already initialized")
        # threads_per_worker must be one, because atari-env is not thread-safe.
        # And because lower the thread-count from the default, we must increase the number of workers
        cls._cluster = LocalCluster(processes=True, asynchronous=False, threads_per_worker=1,
                                    silence_logs=worker_log_level,
                                    n_workers=n_workers,
                                    interface='lo')
        cls._client = Client(cls._cluster)
        cls._client.register_worker_plugin(_CreatorPlugin(class_cb, brain_class), name='creator-plugin')
        logging.info("dask-dashboard available at port: " + str(cls._client.scheduler_info()['services']['dashboard']))

    @classmethod
    def stop_dask(cls):
        cls._client.shutdown()

    @classmethod
    def dask_map(cls, *args, **kwargs):
        if not cls._client:
            raise RuntimeError("dask-client not initialized. Call \"init_dask\" before calling \"dask_map\"")
        return cls._client.gather(cls._client.map(*args, **kwargs))

    @classmethod
    def get_current_worker(cls):
        return get_worker()
