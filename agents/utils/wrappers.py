import gym
import numpy as np


class DelayedObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, delay=1):
        super().__init__(env)
        self.delay = delay
        self._obs_buffer = [np.zeros(self.env.observation_space.shape)] * delay

    def reset(self):
        self._obs_buffer = [np.zeros(self.env.observation_space.shape)] * self.delay
        return self.env.reset()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._obs_buffer.append(obs)
        obs = self._obs_buffer.pop(0)
        return obs, reward, done, info


class ObsWithActionWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            shape=(
                self.env.observation_space.shape[0] + self.env.action_space.shape[0],
            ),
            high=1,
            low=-1,
            dtype=np.float32,
        )

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return np.concatenate([obs, action]), reward, done, info

    def reset(self):
        return np.concatenate([self.env.reset(), np.zeros(self.env.action_space.shape)])


class FrameStack(gym.ObservationWrapper):
    def __init__(self, env, stack_size=4):
        super().__init__(env)
        self.stack_size = stack_size
        self.observation_space = gym.spaces.Box(
            shape=(
                self.env.observation_space.shape[0] * self.stack_size,
            ),
            high=1,
            low=-1,
            dtype=np.float32,
        )
        self._obs_buffer = [np.zeros(self.env.observation_space.shape)] * stack_size

    def reset(self):
        self._obs_buffer = [np.zeros(self.env.observation_space.shape)] * self.stack_size
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        self._obs_buffer.pop(0)
        obs = np.concatenate(self._obs_buffer)
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._obs_buffer.append(obs)
        self._obs_buffer.pop(0)
        obs = np.concatenate(self._obs_buffer)
        return obs, reward, done, info