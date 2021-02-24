import os
import argparse
import collections

import gym
import ptan
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp
from tensorboardX import SummaryWriter

import rc_gym

ENV = 'VSSDetGkDef-v0'
PROCESSES_COUNT = 5
LEARNING_RATE = 0.0001
REPLAY_SIZE = 5000000
REPLAY_INITIAL = 300000
BATCH_SIZE = 256
GAMMA = 0.95
REWARD_STEPS = 2
SAVE_FREQUENCY = 20000
# Q_SIZE = 1


class DDPGActor(nn.Module):
    def __init__(self, obs_size, act_size):
        super(DDPGActor, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_size, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, act_size),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)


class DDPGCritic(nn.Module):
    def __init__(self, obs_size, act_size):
        super(DDPGCritic, self).__init__()

        self.obs_net = nn.Sequential(
            nn.Linear(obs_size, 400),
            nn.ReLU(),
        )

        self.out_net = nn.Sequential(
            nn.Linear(400 + act_size, 300),
            nn.ReLU(),
            nn.Linear(300, 1)
        )

    def forward(self, x, a):
        obs = self.obs_net(x)
        return self.out_net(torch.cat([obs, a], dim=1))


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


def unpack_batch_ddqn(batch, device="cpu"):
    states, actions, rewards, dones, last_states = [], [], [], [], []
    for exp in batch:
        states.append(exp.state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        dones.append(exp.last_state is None)
        if exp.last_state is None:
            last_states.append(exp.state)
        else:
            last_states.append(exp.last_state)
    states_v = ptan.agent.float32_preprocessor(states).to(device)
    actions_v = ptan.agent.float32_preprocessor(actions).to(device)
    rewards_v = ptan.agent.float32_preprocessor(rewards).to(device)
    last_states_v = ptan.agent.float32_preprocessor(last_states).to(device)
    dones_t = torch.BoolTensor(dones).to(device)
    return states_v, actions_v, rewards_v, dones_t, last_states_v


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
    penalties = 0
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
                penalties += extra['penalties']
                break
    return rewards / count, goal_score / count, penalties / count


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

    act_net = DDPGActor(
        test_env.observation_space.shape[0],
        test_env.action_space.shape[0]).to(device)
    act_net.share_memory()
    crt_net = DDPGCritic(
        test_env.observation_space.shape[0],
        test_env.action_space.shape[0]).to(device)

    # Playing
    train_queue = mp.Queue(maxsize=BATCH_SIZE)
    data_proc_list = []
    for _ in range(PROCESSES_COUNT):
        data_proc = mp.Process(target=data_func,
                               args=(act_net, device, train_queue))
        data_proc.start()
        data_proc_list.append(data_proc)

    # Training
    writer = SummaryWriter(comment="-ddpg_" + args.name)
    save_path = os.path.join("saves", "ddpg-" + args.name)
    os.makedirs(save_path, exist_ok=True)

    tgt_act_net = ptan.agent.TargetNet(act_net)
    tgt_crt_net = ptan.agent.TargetNet(crt_net)
    act_opt = optim.Adam(act_net.parameters(), lr=LEARNING_RATE)
    crt_opt = optim.Adam(crt_net.parameters(), lr=LEARNING_RATE)
    buffer = ExperienceReplayBuffer(buffer_size=REPLAY_SIZE)
    n_iter = 0
    n_samples = 0
    finish_event = mp.Event()
    best_reward = None

    try:
        with ptan.common.utils.RewardTracker(writer) as tracker:
            with ptan.common.utils.TBMeanTracker(
                    writer, 100) as tb_tracker:
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
                    states_v, actions_v, rewards_v, \
                        dones_mask, last_states_v = \
                        unpack_batch_ddqn(batch, device)

                    # train critic
                    crt_opt.zero_grad()
                    q_v = crt_net(states_v, actions_v)
                    last_act_v = tgt_act_net.target_model(
                        last_states_v)
                    q_last_v = tgt_crt_net.target_model(
                        last_states_v, last_act_v)
                    q_last_v[dones_mask] = 0.0
                    q_ref_v = rewards_v.unsqueeze(dim=-1) + \
                        q_last_v * GAMMA
                    critic_loss_v = F.mse_loss(
                        q_v, q_ref_v.detach())
                    critic_loss_v.backward()
                    crt_opt.step()
                    tb_tracker.track("loss_critic",
                                     critic_loss_v, n_iter)
                    tb_tracker.track("critic_ref",
                                     q_ref_v.mean(), n_iter)

                    # train actor
                    act_opt.zero_grad()
                    cur_actions_v = act_net(states_v)
                    actor_loss_v = -crt_net(
                        states_v, cur_actions_v)
                    actor_loss_v = actor_loss_v.mean()
                    actor_loss_v.backward()
                    act_opt.step()
                    tb_tracker.track("loss_actor",
                                     actor_loss_v, n_iter)

                    tgt_act_net.alpha_sync(alpha=1 - 1e-3)
                    tgt_crt_net.alpha_sync(alpha=1 - 1e-3)

                    if n_iter % SAVE_FREQUENCY == 0:
                        fname = os.path.join(save_path, "model_act_{}".format(n_iter))
                        torch.save(act_net.state_dict(), fname)
                        fname = os.path.join(save_path, "model_act_latest")
                        torch.save(act_net.state_dict(), fname)
                        fname = os.path.join(save_path, "model_crt_latest")
                        torch.save(crt_net.state_dict(), fname)

                        ts = time.time()
                        rewards, goals_score, penalties = test_net(act_net, test_env, device=device)
                        print("Test done in %.2f sec, reward %.3f, goals_score %d" % (
                            time.time() - ts, rewards, goals_score))
                        writer.add_scalar("test/rewards", rewards, n_iter)
                        writer.add_scalar("test/goals_score", goals_score, n_iter)
                        writer.add_scalar("test/penalties", penalties, n_iter)
                        
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
