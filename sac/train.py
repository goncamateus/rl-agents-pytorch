#!/usr/bin/env python3
import os
import ptan
import gym
import math
import time
import pybullet_envs
import argparse
from tensorboardX import SummaryWriter
import collections
import numpy as np

import torch
import torch.optim as optim
import torch.distributions as distr
import torch.nn as nn
import torch.multiprocessing as mp
import torch.nn.functional as F

import rc_gym

ENV = 'VSS3v3-v0'
PROCESSES_COUNT = 3
LEARNING_RATE = 0.0001
REPLAY_SIZE = 5000000
REPLAY_INITIAL = 256
LR_ACTS = 1e-4
LR_VALS = 1e-4
BATCH_SIZE = 256
GAMMA = 0.95
REWARD_STEPS = 2
SAVE_FREQUENCY = 20000
SAC_ENTROPY_ALPHA = 0.1

class AgentDDPG(ptan.agent.BaseAgent):
    """
    Agent implementing Orstein-Uhlenbeck exploration process
    """

    def __init__(self, net, device="cpu", ou_enabled=True,
                 ou_mu=0.0, ou_teta=0.15, ou_sigma=0.2,
                 ou_epsilon=1.0):
        self.net = net
        self.device = device
        self.ou_enabled = ou_enabled
        self.ou_mu = ou_mu
        self.ou_teta = ou_teta
        self.ou_sigma = ou_sigma
        self.ou_epsilon = ou_epsilon

    def initial_state(self):
        return None

    def __call__(self, states, agent_states):
        states_v = ptan.agent.float32_preprocessor(states)
        states_v = states_v.to(self.device)
        mu_v = self.net(states_v)
        actions = mu_v.data.cpu().numpy()

        if self.ou_enabled and self.ou_epsilon > 0:
            new_a_states = []
            for a_state, action in zip(agent_states, actions):
                if a_state is None:
                    a_state = np.zeros(
                        shape=action.shape, dtype=np.float32)
                a_state += self.ou_teta * (self.ou_mu - a_state)
                a_state += self.ou_sigma * np.random.normal(
                    size=action.shape)

                action += self.ou_epsilon * a_state
                new_a_states.append(a_state)
        else:
            new_a_states = agent_states

        actions = np.clip(actions, -1, 1)
        return actions, new_a_states

class ModelActor(nn.Module):
    def __init__(self, obs_size, act_size):
        super(ModelActor, self).__init__()

        self.mu = nn.Sequential(
            nn.Linear(obs_size, 400),
            nn.Tanh(),
            nn.Linear(400, 300),
            nn.Tanh(),
            nn.Linear(300, act_size),
            nn.Tanh(),
        )
        self.logstd = nn.Parameter(torch.zeros(act_size))

    def forward(self, x):
        return self.mu(x)

class ModelCritic(nn.Module):
    def __init__(self, obs_size):
        super(ModelCritic, self).__init__()

        self.value = nn.Sequential(
            nn.Linear(obs_size, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, 1),
        )

    def forward(self, x):
        return self.value(x)

class ModelSACTwinQ(nn.Module):
    def __init__(self, obs_size, act_size):
        super(ModelSACTwinQ, self).__init__()

        self.q1 = nn.Sequential(
            nn.Linear(obs_size + act_size, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, 1),
        )

        self.q2 = nn.Sequential(
            nn.Linear(obs_size + act_size, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, 1),
        )

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=1)
        return self.q1(x), self.q2(x)

class ExperienceReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = []
        self.capacity = buffer_size
        self.pos = 0

    def __len__(self):
        return len(self.buffer)

    def __iter__(self):
        return iter(self.buffer)

    def sample(self, batch_size):
        if len(self.buffer) <= batch_size:
            return self.buffer
        # Warning: replace=False makes random.choice O(n)
        keys = np.random.choice(len(self.buffer), batch_size, replace=True)
        return [self.buffer[key] for key in keys]

    def add(self, sample):
        if len(self.buffer) < self.capacity:
            self.buffer.append(sample)
        else:
            self.buffer[self.pos] = sample
        self.pos = (self.pos + 1) % self.capacity

class RewardTracker:
    def __init__(self, writer, stop_reward):
        self.writer = writer
        self.stop_reward = stop_reward

    def __enter__(self):
        self.ts = time.time()
        self.ts_frame = 0
        self.total_rewards = []
        return self

    def __exit__(self, *args):
        self.writer.close()

    def reward(self, reward, frame, epsilon=None):
        self.total_rewards.append(reward)
        speed = (frame - self.ts_frame) / (time.time() - self.ts)
        self.ts_frame = frame
        self.ts = time.time()
        mean_reward = np.mean(self.total_rewards[-100:])
        epsilon_str = "" if epsilon is None else ", eps %.2f" % epsilon
        print("%d: done %d games, mean reward %.3f, speed %.2f f/s%s" % (
            frame, len(self.total_rewards), mean_reward, speed, epsilon_str
        ))
        os.system.stdout.flush()
        if epsilon is not None:
            self.writer.add_scalar("epsilon", epsilon, frame)
        self.writer.add_scalar("speed", speed, frame)
        self.writer.add_scalar("reward_100", mean_reward, frame)
        self.writer.add_scalar("reward", reward, frame)
        if mean_reward > self.stop_reward:
            print("Solved in %d frames!" % frame)
            return True
        return False

TotalReward = collections.namedtuple('TotalReward', field_names='reward')


def unpack_batch_a2c(batch, net, last_val_gamma, device="cpu"):
    """
    Convert batch into training tensors
    :param batch:
    :param net:
    :return: states variable, actions tensor, reference values variable
    """
    states = []
    actions = []
    rewards = []
    not_done_idx = []
    last_states = []
    for idx, exp in enumerate(batch):
        states.append(exp.state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        if exp.last_state is not None:
            not_done_idx.append(idx)
            last_states.append(exp.last_state)
    states_v = ptan.agent.float32_preprocessor(states).to(device)
    actions_v = torch.FloatTensor(actions).to(device)

    # handle rewards
    rewards_np = np.array(rewards, dtype=np.float32)
    if not_done_idx:
        last_states_v = ptan.agent.float32_preprocessor(last_states).to(device)
        last_vals_v = net(last_states_v)
        last_vals_np = last_vals_v.data.cpu().numpy()[:, 0]
        rewards_np[not_done_idx] += last_val_gamma * last_vals_np

    ref_vals_v = torch.FloatTensor(rewards_np).to(device)
    return states_v, actions_v, ref_vals_v



@torch.no_grad()
def unpack_batch_sac(batch, val_net, twinq_net, policy_net,
                     gamma: float, ent_alpha: float,
                     device="cpu"):
    """
    Unpack Soft Actor-Critic batch
    """
    states_v, actions_v, ref_q_v = \
        unpack_batch_a2c(batch, val_net, gamma, device)

    # references for the critic network
    mu_v = policy_net(states_v)
    act_dist = distr.Normal(mu_v, torch.exp(policy_net.logstd))
    acts_v = act_dist.sample()
    q1_v, q2_v = twinq_net(states_v, acts_v)
    # element-wise minimum
    ref_vals_v = torch.min(q1_v, q2_v).squeeze() - \
                 ent_alpha * act_dist.log_prob(acts_v).sum(dim=1)
    return states_v, actions_v, ref_vals_v, ref_q_v

def data_func(act_net, device, train_queue):
    env = gym.make(ENV)
    agent = AgentDDPG(act_net, device=device)
    exp_source = ptan.experience.ExperienceSourceFirstLast(
        env, agent, gamma=GAMMA, steps_count=REWARD_STEPS)

    for exp in exp_source:
        new_rewards = exp_source.pop_total_rewards()
        if new_rewards:
            data = TotalReward(reward=np.mean(new_rewards))
            train_queue.put(data)

        train_queue.put(exp)

def test_net(net, env, count=3, device="cpu"):
    rewards = 0.0
    goal_score = 0
    for _ in range(count):
        obs = env.reset()
        while True:
            obs_v = ptan.agent.float32_preprocessor([obs]).to(device)
            mu_v = net(obs_v)
            action = mu_v.squeeze(dim=0).data.cpu().numpy()
            action = np.clip(action, -1, 1)

            obs, reward, done, extra = env.step(action)
            # env.env.render()
            rewards += reward
            if done:
                goal_score += extra['goal_score']
                break
    return rewards / count, goal_score / count

if __name__ == "__main__":
    mp.set_start_method('spawn')
    os.environ['OMP_NUM_THREADS'] = "1"
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False,
                        action="store_true", help="Enable cuda")
    parser.add_argument("-n", "--name", required=True,
                        help="Name of the run")
    args = parser.parse_args()
    device = "cuda" if args.cuda else "cpu"


    test_env = gym.make(ENV)

    act_net = ModelActor(
        test_env.observation_space.shape[0],
        test_env.action_space.shape[0]).to(device)
    crt_net = ModelCritic(
        test_env.observation_space.shape[0]
    ).to(device)
    twinq_net = ModelSACTwinQ(
        test_env.observation_space.shape[0],
        test_env.action_space.shape[0]).to(device)
    print(act_net)
    print(crt_net)
    print(twinq_net)
    
    # Playing
    train_queue = mp.Queue(maxsize=BATCH_SIZE)
    data_proc_list = []
    for _ in range(PROCESSES_COUNT):
        data_proc = mp.Process(target=data_func,
                               args=(act_net, device, train_queue))
        data_proc.start()
        data_proc_list.append(data_proc)

    # Training
    writer = SummaryWriter(comment="-sac_" + args.name)
    save_path = os.path.join("saves", "sac-" + args.name)
    os.makedirs(save_path, exist_ok=True)

    tgt_crt_net = ptan.agent.TargetNet(crt_net)
    act_opt = optim.Adam(act_net.parameters(), lr=LR_ACTS)
    crt_opt = optim.Adam(crt_net.parameters(), lr=LR_VALS)
    twinq_opt = optim.Adam(twinq_net.parameters(), lr=LR_VALS)
    buffer = ExperienceReplayBuffer(buffer_size=REPLAY_SIZE)
    n_iter = 0
    n_samples = 0
    finish_event = mp.Event()
    best_reward = None
    
    try:
        with ptan.common.utils.RewardTracker(writer) as tracker:
            with ptan.common.utils.TBMeanTracker(
                    writer, batch_size=10) as tb_tracker:
                while True:
                    
                    for i in range(BATCH_SIZE):
                        if train_queue.empty():
                            break
                        exp = train_queue.get()

                        if isinstance(exp, TotalReward):
                            tracker.reward(exp.reward, n_samples)
                            continue

                        buffer.add(exp)
                        n_samples += 1
                    
                    if len(buffer) < REPLAY_INITIAL:
                        continue

                    batch = buffer.sample(BATCH_SIZE)
                    states_v, actions_v, ref_vals_v, ref_q_v = \
                        unpack_batch_sac(
                            batch, tgt_crt_net.target_model,
                            twinq_net, act_net, GAMMA,
                            SAC_ENTROPY_ALPHA, device)

                    tb_tracker.track("ref_v", ref_vals_v.mean(), n_iter)
                    tb_tracker.track("ref_q", ref_q_v.mean(), n_iter)

                    # train TwinQ
                    twinq_opt.zero_grad()
                    q1_v, q2_v = twinq_net(states_v, actions_v)
                    q1_loss_v = F.mse_loss(q1_v.squeeze(),
                                        ref_q_v.detach())
                    q2_loss_v = F.mse_loss(q2_v.squeeze(),
                                        ref_q_v.detach())
                    q_loss_v = q1_loss_v + q2_loss_v
                    q_loss_v.backward()
                    twinq_opt.step()
                    tb_tracker.track("loss_q1", q1_loss_v, n_iter)
                    tb_tracker.track("loss_q2", q2_loss_v, n_iter)

                    # Critic
                    crt_opt.zero_grad()
                    val_v = crt_net(states_v)
                    v_loss_v = F.mse_loss(val_v.squeeze(),
                                        ref_vals_v.detach())
                    v_loss_v.backward()
                    crt_opt.step()
                    tb_tracker.track("loss_v", v_loss_v, n_iter)

                    # Actor
                    act_opt.zero_grad()
                    acts_v = act_net(states_v)
                    q_out_v, _ = twinq_net(states_v, acts_v)
                    act_loss = -q_out_v.mean()
                    act_loss.backward()
                    act_opt.step()
                    tb_tracker.track("loss_act", act_loss, n_iter)

                    tgt_crt_net.alpha_sync(alpha=1 - 1e-3)

                    if n_iter % SAVE_FREQUENCY == 0:
                        fname = os.path.join(save_path, "model_act_{}".format(n_iter))
                        torch.save(act_net.state_dict(), fname)
                        fname = os.path.join(save_path, "model_act_latest")
                        torch.save(act_net.state_dict(), fname)
                        fname = os.path.join(save_path, "model_crt_latest")
                        torch.save(crt_net.state_dict(), fname)

                        ts = time.time()
                        rewards, goals_score = test_net(act_net, test_env, device=device)
                        print("Test done in %.2f sec, reward %.3f, goals_score %d" % (
                            time.time() - ts, rewards, goals_score))
                        writer.add_scalar("test_rewards", rewards, n_iter)
                        writer.add_scalar("test_goals_score", goals_score, n_iter)
                        
                        if best_reward is None or best_reward < rewards:
                            if best_reward is not None:
                                print("Best reward updated: %.3f -> %.3f" % (best_reward, rewards))
                                name = "best_%+.3f_%d.dat" % (rewards, n_iter)
                                fname = os.path.join(save_path, "model_act_best")
                                torch.save(act_net.state_dict(), fname)
                            best_reward = rewards
                    
                    n_iter += 1
                    
                    tb_tracker.track("sample/train ratio",
                                        (n_samples/n_iter), n_iter)

    except KeyboardInterrupt:
        print("...Finishing...")
        finish_event.set()

    finally:
        if train_queue:
            while train_queue.qsize() > 0:
                train_queue.get()

        print('queue is empty')

        print("Waiting for threads to finish...")
        for p in data_proc_list:
            p.terminate()
            p.join()

        del(train_queue)
        del(act_net)