import collections
import copy
import os
import time
from dataclasses import dataclass

import gym
import numpy as np
import torch
from agents.utils import (DelayedObservationWrapper, ObsWithActionWrapper,ExperienceFirstLast,
                          HyperParameters, OrnsteinUhlenbeckNoise,
                          generate_gif, FrameStack)


@dataclass
class DDPGHP(HyperParameters):
    AGENT: str = "ddpg_async"
    NOISE_SIGMA_INITIAL: float = None  # Initial action noise sigma
    NOISE_THETA: float = None
    NOISE_SIGMA_DECAY: float = None  # Action noise sigma decay
    NOISE_SIGMA_MIN: float = None
    NOISE_SIGMA_GRAD_STEPS: float = None  # Decay action noise every _ grad steps
    N_REWARDS: int = 1  # Number of rewards


def data_func(pi, device, queue_m, finish_event_m, sigma_m, gif_req_m, hp):
    env = gym.make(hp.ENV_NAME)
    env = ObsWithActionWrapper(env)
    env = FrameStack(env, hp.STACK_SIZE)
    env = DelayedObservationWrapper(env, delay=hp.DELAY)
    noise = OrnsteinUhlenbeckNoise(
        sigma=sigma_m.value,
        theta=hp.NOISE_THETA,
        min_value=env.action_space.low,
        max_value=env.action_space.high,
    )

    with torch.no_grad():
        while not finish_event_m.is_set():
            # Check for generate gif request
            gif_idx = -1
            with gif_req_m.get_lock():
                if gif_req_m.value != -1:
                    gif_idx = gif_req_m.value
                    gif_req_m.value = -1
            if gif_idx != -1:
                path = os.path.join(hp.GIF_PATH, f"{gif_idx:09d}.gif")
                generate_gif(env=env, filepath=path, pi=copy.deepcopy(pi), hp=hp)

            done = False
            s = env.reset()
            noise.reset()
            noise.sigma = sigma_m.value
            info = {}
            ep_steps = 0
            if hp.MULTI_AGENT:
                ep_rw = [0] * hp.N_AGENTS
            else:
                ep_rw = 0
            st_time = time.perf_counter()
            for i in range(hp.MAX_EPISODE_STEPS):
                # Step the environment
                s_v = torch.Tensor(s).to(device)
                a_v = pi(s_v)
                a = a_v.cpu().numpy()
                a = noise(a)
                s_next, r, done, info = env.step(a)     
                ep_steps += 1
                if hp.MULTI_AGENT:
                    for i in range(hp.N_AGENTS):
                        ep_rw[i] += r[f"robot_{i}"]
                else:
                    if hp.N_REWARDS > 1:
                        ep_rw += r.sum()
                    else:
                        ep_rw += r

                # Trace NStep rewards and add to mp queue
                if hp.MULTI_AGENT:
                    exp = list()
                    for i in range(hp.N_AGENTS):
                        if done:
                            s_next[i] = None
                        kwargs = {
                            "state": s[i],
                            "action": a[i],
                            "reward": r[f"robot_{i}"],
                            "last_state": s_next[i],
                        }
                        exp.append(ExperienceFirstLast(**kwargs))
                    queue_m.put(exp)
                else:
                    if done:
                        s_next = None
                    kwargs = {
                        "state": s,
                        "action": a,
                        "reward": r*100,
                        "last_state": s_next,
                    }
                    exp = ExperienceFirstLast(**kwargs)
                    queue_m.put(exp)

                if done:
                    break

                # Set state for next step
                s = s_next

            info["fps"] = ep_steps / (time.perf_counter() - st_time)
            info["noise"] = noise.sigma
            info["ep_steps"] = ep_steps
            info["ep_rw"] = ep_rw
            queue_m.put(info)
