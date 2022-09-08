import gym
import rsoccer_gym
import numpy as np
from agents.ddpg import DDPGActor
import torch

env = gym.make('VSS5v5-v0')

obs = env.reset()

print(env.action_space.shape)

model = DDPGActor(11, 2)
model.load_state_dict(torch.load('checkpoint_000600000.pth')['pi_state_dict'])

obs = torch.FloatTensor(obs)

traced_script_module = torch.jit.trace(model, obs)

torch.jit.save(traced_script_module, 'atk_gonca.pt')