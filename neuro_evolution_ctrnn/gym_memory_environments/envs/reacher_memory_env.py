import numpy as np
from gym.envs.mujoco import ReacherEnv
from typing import Tuple


class ReacherMemoryEnv(ReacherEnv):

    def __init__(self, observation_frames: int, memory_frames: int, action_frames: int):
        assert observation_frames > 0 and memory_frames > 0 and action_frames > 0

        self.observation_frames = observation_frames
        self.memory_frames = memory_frames
        self.action_frames = action_frames
        self.observation_mask = [4, 5, 8, 9, 10]
        self.t = 0
        self._max_episode_steps = self.observation_frames + self.memory_frames + self.action_frames
        super().__init__()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool, np.ndarray]:

        if self.t <= self.observation_frames + self.memory_frames:
            action = np.zeros(action.shape)

        ob, rew, done, info = super().step(action)

        if self.t >= self.observation_frames:
            for index in self.observation_mask:
                ob[index] = 0.0

        if self.t < self.observation_frames + self.memory_frames:
            rew = 0.0

        if self.t >= self._max_episode_steps:
            done = True

        self.t += 1
        return ob, rew, done, info

    def reset(self):
        ob = super().reset()
        self._max_episode_steps = self.observation_frames + self.memory_frames + self.action_frames
        self.t = 0

        return ob
