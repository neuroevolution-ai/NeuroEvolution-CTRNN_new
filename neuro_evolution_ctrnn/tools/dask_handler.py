from dask.distributed import Client, Worker, WorkerPlugin
from typing import Optional
import logging
import gym
from dask.distributed import get_worker

get_current_worker = get_worker


class EnvPlugin(WorkerPlugin):
    """
    This WorkerPlugin initialized a gym-object and binds it to the worker whenever a new worker gets started

    """

    def __init__(self, env_id):
        self.env_id = env_id

    def setup(self, worker: Worker):
        # called exactly once for every worker before it executes the first task
        if self.env_id:
            worker.env = gym.make(self.env_id)
            logging.info("creating new env for worker: " + str(worker))

    def teardown(self, worker: Worker):
        pass

    def transition(self, key: str, start: str, finish: str, **kwargs):
        # called whenever worker gets new task, i think
        pass


class DaskHandler:
    _client: Optional[Client] = None

    @classmethod
    def init_dask(cls):
        if cls._client:
            raise RuntimeError("dask client already initialized")
        cls._client: Client = Client(processes=True, asynchronous=False)

    @classmethod
    def init_workers_with_env(cls, env_id: str):
        cls._client.register_worker_plugin(EnvPlugin(env_id), name='env-plugin')

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
