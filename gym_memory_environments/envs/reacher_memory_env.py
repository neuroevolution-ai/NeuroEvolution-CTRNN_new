import numpy as np
from gym.envs.mujoco import ReacherEnv


class ReacherMemoryEnv(ReacherEnv):

    def __init__(self, observation_frames, memory_frames, action_frames, observation_mask):
        assert observation_frames > 0 and memory_frames > 0 and action_frames > 0

        self.observation_frames = observation_frames
        self.memory_frames = memory_frames
        self.action_frames = action_frames
        self.observation_mask = observation_mask

        self.t = 0

        super().__init__()

    def step(self, action):

        if self.t <= self.observation_frames + self.memory_frames:
            action = np.zeros(action.shape)

        ob, rew, done, info = super().step(action)

        if self.t >= self.observation_frames:
            for index in self.observation_mask:
                ob[index] = 0.0

        if self.t < self.observation_frames + self.memory_frames:
            rew = 0.0

        self.t += 1

        return ob, rew, done, info

    def reset(self):
        super().reset()
        self.spec.max_episode_steps = self.observation_frames + self.memory_frames + self.action_frames
        self.t = 0
