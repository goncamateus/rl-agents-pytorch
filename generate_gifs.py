import argparse
import os

import gym
import numpy as np
import rsoccer_gym
import torch

from agents.ddpg import DDPGActor
from agents.sac import GaussianPolicy
from agents.utils.gif import generate_gif
from agents.utils.wrappers import FrameStack, ObsWithActionWrapper


def get_env_specs(env_name):
    env = gym.make(env_name)
    return (
        env.observation_space.shape[0],
        env.action_space.shape[0],
        env.spec.max_episode_steps,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cuda", default=False, action="store_true", help="Enable cuda"
    )
    parser.add_argument("-c", "--checkpoint", required=True, help="checkpoint to load")
    args = parser.parse_args()
    device = "cuda" if args.cuda else "cpu"

    checkpoint = torch.load(args.checkpoint)

    env = gym.make(checkpoint["ENV_NAME"])
    env = ObsWithActionWrapper(env)
    env = FrameStack(env, 50)

    if checkpoint["AGENT"] == "ddpg_async":
        pi = DDPGActor(checkpoint["N_OBS"], checkpoint["N_ACTS"]).to(device)
    elif checkpoint["AGENT"] == "sac_async":
        pi = GaussianPolicy(
            checkpoint["N_OBS"],
            checkpoint["N_ACTS"],
            checkpoint["LOG_SIG_MIN"],
            checkpoint["LOG_SIG_MAX"],
            checkpoint["EPSILON"],
        ).to(device)
    else:
        raise AssertionError

    pi.load_state_dict(checkpoint["pi_state_dict"])
    pi.eval()
    # obs = env.reset()
    # obs = torch.Tensor(obs).to(device)
    # traced_script_module = torch.jit.trace(pi, obs)
    # torch.jit.save(traced_script_module, "atk.pt")

    generate_gif(
        env=env,
        filepath=args.checkpoint.replace("pth", "gif").replace("checkpoint", "gif"),
        pi=pi,
        hp=checkpoint,
    )
