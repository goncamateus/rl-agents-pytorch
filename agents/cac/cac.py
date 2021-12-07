import copy
import os
import time
from dataclasses import dataclass

import gym
import torch
from agents.ddpg.networks import DDPGActor, DDPGCritic, TargetActor, TargetCritic
from agents.utils.experience import ExperienceFirstLastCAC
from agents.utils.experiment import HyperParameters
from agents.utils.gif import generate_gif
from agents.utils.noise import OrnsteinUhlenbeckNoise
from torch.nn import functional as F
from torch.optim import Adam


@dataclass
class CACHP(HyperParameters):
    AGENT: str = "cac_async"
    NOISE_SIGMA_INITIAL: float = None  # Initial action noise sigma
    NOISE_THETA: float = None
    NOISE_SIGMA_DECAY: float = None  # Action noise sigma decay
    NOISE_SIGMA_MIN: float = None
    NOISE_SIGMA_GRAD_STEPS: float = None  # Decay action noise every _ grad steps
    N_REWARDS: int = 1  # Number of rewards


def data_func(agent, device, queue_m, finish_event_m, sigma_m, gif_req_m, hp):
    env = gym.make(hp.ENV_NAME)
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
                generate_gif(env=env, filepath=path, pi=copy.deepcopy(agent), hp=hp)

            done = False
            s = env.reset()
            noise.reset()
            noise.sigma = sigma_m.value
            info = {}
            ep_steps = 0
            ep_rw = 0
            st_time = time.perf_counter()
            for i in range(hp.MAX_EPISODE_STEPS):
                # Step the environment
                actions = agent(s)
                a = actions[0]
                council = actions[1:]
                a = noise(a)
                s_next, r, done, info = env.step(a)
                ep_steps += 1
                ep_rw += r.sum()
                if done:
                    s_next = None
                kwargs = {
                    "state": s,
                    "action": a,
                    "council": council,
                    "reward": r,
                    "last_state": s_next,
                }
                exp = ExperienceFirstLastCAC(**kwargs)
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


class CAC:
    def __init__(self, hp: CACHP):
        self.hp = hp
        self.device = torch.device(self.hp.DEVICE)
        # Actor-Critic
        self.pi = DDPGActor(hp.N_ACTS * (hp.N_REWARDS - 1), hp.N_ACTS).to(self.device)
        self.pi_council = [
            DDPGActor(hp.N_OBS, hp.N_ACTS).to(self.device)
            for _ in range(hp.N_REWARDS - 1)
        ]
        self.Q = DDPGCritic(hp.N_OBS, hp.N_ACTS).to(self.device)
        self.Q_council = [
            DDPGCritic(hp.N_OBS, hp.N_ACTS).to(self.device)
            for _ in range(hp.N_REWARDS - 1)
        ]
        # Training
        self.tgt_Q = TargetCritic(self.Q)
        self.tgt_council_Q = [TargetCritic(critic) for critic in self.Q_council]
        self.tgt_pi = TargetActor(self.pi)
        self.tgt_council_pi = [TargetActor(actor) for actor in self.pi_council]
        self.pi_opt = Adam(self.pi.parameters(), lr=hp.LEARNING_RATE)
        self.pi_council_opt = [
            Adam(actor.parameters(), lr=hp.LEARNING_RATE) for actor in self.pi_council
        ]
        self.Q_opt = Adam(self.Q.parameters(), lr=hp.LEARNING_RATE)
        self.Q_council_opt = [
            Adam(critic.parameters(), lr=hp.LEARNING_RATE) for critic in self.Q_council
        ]

        self.gamma = hp.GAMMA ** hp.REWARD_STEPS

    def __call__(self, observation):
        s_v = torch.Tensor(observation).to(self.device)
        councils = [actor(s_v) for actor in self.pi_council]
        councils_v = torch.cat(councils)
        action = self.pi.get_action(councils_v)
        actions = [action]
        for council in councils:
            actions.append(council.detach().cpu().numpy())
        return actions

    def get_action(self, observation):
        s_v = torch.Tensor(observation).to(self.device)
        council = [actor(s_v) for actor in self.pi_council]
        council = torch.cat(council)
        return self.pi.get_action(council)

    def share_memory(self):
        self.pi.share_memory()
        self.Q.share_memory()
        for i in range(self.hp.N_REWARDS -1 ):
            self.pi_council[i].share_memory()
            self.Q_council[i].share_memory()

    def train(self, batch):
        metrics = {}
        alphas = torch.Tensor([0.5, 0.48732, 0.00713, 0.00509]).to(self.device)
        S_v = batch[0]
        A_v = batch[1]
        r_v = batch[2]*alphas*100
        S_next_v = batch[3]
        dones = batch[4]
        A_council_v = batch[5]

        # Train Council
        for i in range(self.hp.N_REWARDS - 1):
            # Train Critic
            Qs = self.Q_council[i](S_v, A_council_v[i])
            A_next = self.tgt_council_pi[i](S_next_v)
            Q_next = self.tgt_council_Q[i](S_next_v, A_next)
            Q_next[dones == 1.0] = 0.0
            Q_next = r_v[:, i].unsqueeze(1) + self.hp.GAMMA * Q_next
            Q_next = Q_next.detach()
            self.Q_council_opt[i].zero_grad()
            Q_council_loss = F.mse_loss(Qs, Q_next)
            Q_council_loss.backward()
            self.Q_council_opt[i].step()

            # Train Actor
            self.pi_council_opt[i].zero_grad()
            Actions = self.pi_council[i](S_v)
            Qs = self.Q_council[i](S_v, Actions)
            pi_council_loss = -Qs.mean()
            pi_council_loss.backward()
            self.pi_council_opt[i].step()

            metrics[f"loss_Q_{i}"] = Q_council_loss.item()
            metrics[f"loss_pi_{i}"] = pi_council_loss.item()

        # Train Sparse Networks
        # Train Critic
        Qs = self.Q(S_v, A_v)
        council_actions = [actor(S_next_v) for actor in self.pi_council]
        council_actions = torch.cat(council_actions, dim=1)
        A_next = self.tgt_pi(council_actions)
        Q_next = self.tgt_Q(S_next_v, A_next)
        for council in self.Q_council:
            Q_next += council(S_next_v, A_next)
        Q_next[dones == 1.0] = 0.0
        Q_next = r_v.sum(dim=1).unsqueeze(1) + self.gamma * Q_next
        Q_next = Q_next.detach()
        self.Q_opt.zero_grad()
        Q_loss = F.mse_loss(Qs, Q_next)
        Q_loss.backward()
        self.Q_opt.step()

        # Train Actor
        self.pi_opt.zero_grad()
        council_actions = torch.cat(A_council_v, dim=1)
        Actions = self.pi(council_actions)
        Qs = self.Q(S_v, Actions)
        pi_loss = -Qs.mean()
        pi_loss.backward()
        self.pi_opt.step()
        metrics["loss_Q"] = Q_loss.item()
        metrics["loss_pi"] = pi_loss.item()

        # Sync target networks
        self.tgt_pi.sync(alpha=1 - 1e-3)
        self.tgt_Q.sync(alpha=1 - 1e-3)
        for i in range(self.hp.N_REWARDS - 1):
            self.tgt_council_pi[i].sync(alpha=1 - 1e-3)
            self.tgt_council_Q[i].sync(alpha=1 - 1e-3)
        return metrics
