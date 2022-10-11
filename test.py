import gym
import rsoccer_gym
import numpy as np

from agents.ddpg import DDPGActor
from agents.utils.wrappers import FrameStack, ObsWithActionWrapper

import torch

env = gym.make('VSS-v0')
env = ObsWithActionWrapper(env)
env = FrameStack(env, 10)

obs = env.reset()

# print(env.action_space.shape)
model = DDPGActor(obs.shape[0], env.action_space.shape[0])
model.load_state_dict(torch.load('checkpoint_000300000.pth')['pi_state_dict'])

obs = torch.FloatTensor(obs)

traced_script_module = torch.jit.trace(model, obs)

torch.jit.save(traced_script_module, 'atk-3.pt')