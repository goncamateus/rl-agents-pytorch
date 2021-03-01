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

from experience import ExperienceSourceFirstLast

import tracemalloc
tracemalloc.start()


ENV = 'VSS1v1SelfPlay-v0'
PROCESSES_COUNT = 2
LEARNING_RATE = 0.0001
LEARNING_RATE_ACT = 0.0001
REPLAY_SIZE = 1000000
# REPLAY_INITIAL = 100000
REPLAY_INITIAL = 100000
CRITIC_INITIAL = 100000
BATCH_SIZE = 256
GAMMA = 0.95
REWARD_STEPS = 2
SAVE_FREQUENCY = 50000
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


TotalReward = collections.namedtuple('TotalReward', ['reward', 'move', 'energy', 'goals','goal_score', 'ball_grad'])

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

def test_net(net_1, net_2, env, count=1, device="cpu"):
    rewards = 0.0
    steps = 0
    for _ in range(count):
        obs = env.reset()
        while True:
            obs_v = ptan.agent.float32_preprocessor([obs[0]]).to(device)
            mu_v = net_1(obs_v)
            action = mu_v.squeeze(dim=0).data.cpu().numpy()
            action_1 = np.clip(action, -1, 1)
            
            obs_v = ptan.agent.float32_preprocessor([obs[1]]).to(device)
            mu_v = net_2(obs_v)
            action = mu_v.squeeze(dim=0).data.cpu().numpy()
            action_2 = np.clip(action, -1, 1)

            obs, reward, done, extra = env.step([action_1, action_2])
            env.env.render()
            # rewards += reward[0]
            steps += 1
            if done[0]:
                rewards += extra[0]['goal_score']
                break
    return rewards / count, steps / count

def data_func(act_net, device, train_queue):
    env = gym.make(ENV)
    agent = AgentDDPG(act_net, device=device)
    exp_source = ExperienceSourceFirstLast(
        env, agent, gamma=GAMMA, steps_count=REWARD_STEPS, vectorized=True)

    for exp in exp_source:
        new_rewards = exp_source.pop_rewards_extras()
        if new_rewards:
            data = TotalReward(reward=np.mean(new_rewards[0]), move=np.mean(new_rewards[1]), energy=np.mean(new_rewards[2]), goals=np.mean(new_rewards[3]),goal_score=np.mean(new_rewards[4]), ball_grad=np.mean(new_rewards[5]))
            train_queue.put(data)

        train_queue.put(exp)


if __name__ == "__main__":
    mp.set_start_method('spawn')
    os.environ['OMP_NUM_THREADS'] = "1"
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False,
                        action="store_true", help="Enable cuda")
    parser.add_argument("-n", "--name", required=True,
                        help="Name of the run")
    parser.add_argument("-m", "--model", required=False, help="Model file to load")
    args = parser.parse_args()
    device = "cuda" if args.cuda else "cpu"


    act_net = DDPGActor(40,2).to(device)
    # act_net.load_state_dict(torch.load('/home/felipe/Documents/rl-agents-pytorch/ddpg_selfplay/saves/ddpg-self_play_rwshaping_v2ms_a1_27/model_act_latest'))
    init_net = DDPGActor(40,2).to(device)
    if args.model:
        init_net.load_state_dict(torch.load(args.model))
    act_net.share_memory()
    crt_net = DDPGCritic(40,2).to(device)
    # crt_net.load_state_dict(torch.load('/home/felipe/Documents/rl-agents-pytorch/ddpg_selfplay/saves/ddpg-self_play_rwshaping_v2ms_a1_27/model_crt_latest'))


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
    act_opt = optim.Adam(act_net.parameters(), lr=LEARNING_RATE_ACT)
    crt_opt = optim.Adam(crt_net.parameters(), lr=LEARNING_RATE)
    buffer = ExperienceReplayBuffer(buffer_size=REPLAY_SIZE)
    n_iter = 0
    n_samples = 0
    finish_event = mp.Event()
    test_env = gym.make(ENV)
    best_reward = None
    last_gpu_alloc = torch.cuda.memory_allocated(device=device)
    last_gpu_reserved = torch.cuda.memory_reserved(device=device)

    snapshot1 = tracemalloc.take_snapshot()
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
                            writer.add_scalar("shaping/move", exp.move, n_samples)
                            writer.add_scalar("shaping/energy", exp.energy, n_samples)
                            writer.add_scalar("shaping/goals", exp.goals, n_samples)
                            writer.add_scalar("shaping/goal_score", exp.goal_score, n_samples)
                            # writer.add_scalar("shaping/ball_grad", exp.ball_grad, n_samples)
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
                        rewards, steps = test_net(act_net, init_net, test_env, device=device)
                        print("Test done in %.2f sec, reward %.3f, steps %d" % (
                            time.time() - ts, rewards, steps))
                        writer.add_scalar("test_goal_score", rewards, n_iter)
                        if best_reward is None or best_reward < rewards:
                            if best_reward is not None:
                                print("Best reward updated: %.3f -> %.3f" % (best_reward, rewards))
                                name = "best_%+.3f_%d.dat" % (rewards, n_iter)
                                fname = os.path.join(save_path, "model_act_best")
                                torch.save(act_net.state_dict(), fname)
                            best_reward = rewards
                                            
                        # ... call the function leaking memory ...
                        snapshot2 = tracemalloc.take_snapshot()

                        top_stats = snapshot2.compare_to(snapshot1, 'lineno')

                        print("[ Top 10 differences ]")
                        for stat in top_stats[:10]:
                            print(stat)
                            
                        print("memory allocated delta: ",torch.cuda.memory_allocated(device=device) - last_gpu_alloc)
                        print("memory reserved delta: ",torch.cuda.memory_reserved(device=device) -last_gpu_reserved)
                        snapshot1 = snapshot2
                        last_gpu_alloc = torch.cuda.memory_allocated(device=device)
                        last_gpu_reserved = torch.cuda.memory_reserved(device=device)

                    n_iter += 1

                    tb_tracker.track("sample/train ratio",
                                     (n_samples/n_iter), n_iter)



    except KeyboardInterrupt:
        print("...Finishing...")
        finish_event.set()

    # except Exception:
    #     print("!!! Exception caught on main !!!")
    #     finish_event.set()

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
