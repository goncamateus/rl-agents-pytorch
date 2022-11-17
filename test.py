import gym
import rsoccer_gym
import numpy as np

from agents.ddpg import DDPGActor
from agents.utils.wrappers import FrameStack, ObsWithActionWrapper, SkipFrameWrapper

import torch

env = gym.make('VSS-v0')
env = ObsWithActionWrapper(env)
env = FrameStack(env, 10)

obs = env.reset()

# print(env.action_space.shape)
model = DDPGActor(obs.shape[0], env.action_space.shape[0])
model.load_state_dict(torch.load('/home/gonca/rl/rl-agents-pytorch/saves/VSS-v0/ddpg_async/WithAction-Skip3-Stack10-step0.01/checkpoints/checkpoint_000550000.pth')['pi_state_dict'])

obs = torch.FloatTensor(obs)
print(model)

traced_script_module = torch.jit.trace(model, obs)

torch.jit.save(traced_script_module, 'WithAction-Skip3-Stack10-step0.01.pt')