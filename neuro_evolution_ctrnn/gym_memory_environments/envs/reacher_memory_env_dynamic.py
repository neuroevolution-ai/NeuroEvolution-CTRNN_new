"""Modification of ReacherMemoryEnv inspired by openAi's Algorithmic environments, which become more
dificualt as agents become better"""

from . import ReacherMemoryEnv
import logging


class ReacherMemoryEnvDynamic(ReacherMemoryEnv):
    MIN_REWARD_SHORTFALL_FOR_PROMOTION = -6.0

    def __init__(self, observation_frames: int, memory_frames: int, action_frames: int):
        self.episode_total_reward = 0.0
        super(ReacherMemoryEnvDynamic, self).__init__(observation_frames=observation_frames,
                                                      memory_frames=memory_frames,
                                                      action_frames=action_frames)
        self.reward_shortfalls = []
        self.last = 100
        self.memory_frames_min = memory_frames

    def step(self, action):
        ob, rew, done, info = super(ReacherMemoryEnvDynamic, self).step(action)
        self.episode_total_reward += rew
        return ob, rew, done, info

    def _check_levelup(self):
        """Called between episodes. Update our running record of episode rewards
        and, if appropriate, 'level up' number of memory_frames.

        This is copied from
        https://github.com/openai/gym/blob/bf7e44f680fa/gym/envs/algorithmic/algorithmic_env.py#L205"""
        if self.episode_total_reward is None:
            # This is before the first episode/call to reset(). Nothing to do.
            return
        self.reward_shortfalls.append(self.episode_total_reward)
        self.reward_shortfalls = self.reward_shortfalls[-self.last:]
        if len(self.reward_shortfalls) == self.last and \
                min(self.reward_shortfalls) >= self.MIN_REWARD_SHORTFALL_FOR_PROMOTION and \
                self.memory_frames_min < 40:
            self.memory_frames_min += 1
            self.reward_shortfalls = []
            logging.info("promotion!")

    def reset(self):
        self._check_levelup()
        self.memory_frames = self.np_random.randint(3) + self.memory_frames_min
        ob = super(ReacherMemoryEnvDynamic, self).reset()
        self.episode_total_reward = 0.0
        return ob
