import os
import time
import copy
import queue
import random
import numpy as np
from math import inf

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

from velodyne_env import GazeboEnv


# =========================================================
# Global config
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed = 0
eval_freq = int(2e4)            # 기존 5e3보다 완화
max_ep = 500
eval_ep = 5
max_timesteps = int(5e6)

expl_noise = 1.0
expl_decay_steps = 500000
expl_min = 0.1

batch_size = 40
discount = 0.99999
tau = 0.005
policy_noise = 0.2
noise_clip = 0.5
policy_freq = 2
buffer_size = int(1e6)

file_name = "260309_finetune_parallel_curriculum"
save_model = True
load_model = False
random_near_obstacle = True

environment_dim = 20
robot_dim = 4
state_dim = environment_dim + robot_dim
action_dim = 2
max_action = 1.0

num_workers = 4                 # 병렬 Gazebo worker 개수
warmup_steps = 2000             # replay buffer에 최소한 쌓일 때까지 learner 대기
train_steps_per_env_step = 1    # 환경 step당 learner update 비율
sync_interval = 1000            # learner -> workers actor weight 동기화 주기
queue_timeout = 1.0

# curriculum / finetuning options
load_pretrained = True
freeze_actor_layers = []        # 예: ["layer_1"]
freeze_critic = False
actor_lr_finetune = 1e-4
critic_lr_finetune = 1e-4

pretrained_models = "./pytorch_models"
pretrained_file = "TD3_velodyne"

results_dir = "./finetune_results_parallel"
models_dir = "./finetune_models_parallel"

os.makedirs(results_dir, exist_ok=True)
if save_model:
    os.makedirs(models_dir, exist_ok=True)


# =========================================================
# Utils
# =========================================================
def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)


# =========================================================
# Replay buffer
# =========================================================
class ReplayBuffer(object):
    def __init__(self, max_size, seed):
        self.max_size = int(max_size)
        self.storage = []
        self.ptr = 0
        random.seed(seed)

    def add(self, state, action, reward, done, next_state):
        data = (state, action, reward, done, next_state)

        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample_batch(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)

        states, actions, rewards, dones, next_states = [], [], [], [], []
        for i in ind:
            s, a, r, d, ns = self.storage[i]
            states.append(np.array(s, copy=False))
            actions.append(np.array(a, copy=False))
            rewards.append(np.array([r], copy=False))
            dones.append(np.array([d], copy=False))
            next_states.append(np.array(ns, copy=False))

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(dones),
            np.array(next_states),
        )

    def size(self):
        return len(self.storage)


# =========================================================
# Networks
# =========================================================
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.layer_1 = nn.Linear(state_dim, 800)
        self.layer_2 = nn.Linear(800, 600)
        self.layer_3 = nn.Linear(600, action_dim)
        self.tanh = nn.Tanh()

    def forward(self, s):
        s = F.relu(self.layer_1(s))
        s = F.relu(self.layer_2(s))
        a = self.tanh(self.layer_3(s))
        return a


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.layer_1 = nn.Linear(state_dim, 800)
        self.layer_2_s = nn.Linear(800, 600)
        self.layer_2_a = nn.Linear(action_dim, 600)
        self.layer_3 = nn.Linear(600, 1)

        self.layer_4 = nn.Linear(state_dim, 800)
        self.layer_5_s = nn.Linear(800, 600)
        self.layer_5_a = nn.Linear(action_dim, 600)
        self.layer_6 = nn.Linear(600, 1)

    def forward(self, s, a):
        s1 = F.relu(self.layer_1(s))
        self.layer_2_s(s1)
        self.layer_2_a(a)
        s11 = torch.mm(s1, self.layer_2_s.weight.data.t())
        s12 = torch.mm(a, self.layer_2_a.weight.data.t())
        s1 = F.relu(s11 + s12 + self.layer_2_a.bias.data)
        q1 = self.layer_3(s1)

        s2 = F.relu(self.layer_4(s))
        self.layer_5_s(s2)
        self.layer_5_a(a)
        s21 = torch.mm(s2, self.layer_5_s.weight.data.t())
        s22 = torch.mm(a, self.layer_5_a.weight.data.t())
        s2 = F.relu(s21 + s22 + self.layer_5_a.bias.data)
        q2 = self.layer_6(s2)
        return q1, q2


# =========================================================
# TD3
# =========================================================
class TD3(object):
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

        self.max_action = max_action
        self.writer = SummaryWriter()
        self.iter_count = 0

    def get_action(self, state):
        state = torch.Tensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(
        self,
        replay_buffer,
        iterations,
        batch_size=100,
        discount=1.0,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
    ):
        if replay_buffer.size() < batch_size:
            return

        av_Q = 0
        max_Q = -inf
        av_loss = 0

        for it in range(iterations):
            (
                batch_states,
                batch_actions,
                batch_rewards,
                batch_dones,
                batch_next_states,
            ) = replay_buffer.sample_batch(batch_size)

            state = torch.Tensor(batch_states).to(device)
            next_state = torch.Tensor(batch_next_states).to(device)
            action = torch.Tensor(batch_actions).to(device)
            reward = torch.Tensor(batch_rewards).to(device)
            done = torch.Tensor(batch_dones).to(device)

            next_action = self.actor_target(next_state)

            noise = torch.Tensor(batch_actions).data.normal_(0, policy_noise).to(device)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (next_action + noise).clamp(-self.max_action, self.max_action)

            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            av_Q += torch.mean(target_Q).item()
            max_Q = max(max_Q, torch.max(target_Q).item())
            target_Q = reward + ((1 - done) * discount * target_Q).detach()

            current_Q1, current_Q2 = self.critic(state, action)
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            if self.critic_optimizer is not None:
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

            if it % policy_freq == 0:
                actor_loss = -self.critic(state, self.actor(state))[0].mean()

                if self.actor_optimizer is not None:
                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor_optimizer.step()

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                av_loss += actor_loss.item()

        self.iter_count += 1
        self.writer.add_scalar("train/avg_q", av_Q / max(1, iterations), self.iter_count)
        self.writer.add_scalar("train/max_q", max_Q, self.iter_count)
        self.writer.add_scalar("train/avg_actor_loss", av_loss / max(1, iterations), self.iter_count)

    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), f"{directory}/{filename}_actor.pth")
        torch.save(self.critic.state_dict(), f"{directory}/{filename}_critic.pth")

    def load(self, filename, directory):
        self.actor.load_state_dict(
            torch.load(f"{directory}/{filename}_actor.pth", map_location=device, weights_only=True)
        )
        self.critic.load_state_dict(
            torch.load(f"{directory}/{filename}_critic.pth", map_location=device, weights_only=True)
        )
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())


# =========================================================
# Curriculum helpers
# =========================================================
def load_and_freeze(network, base_filename, directory,
                    freeze_actor_layers=None, freeze_critic=False,
                    device_str="cpu"):
    if freeze_actor_layers is None:
        freeze_actor_layers = []

    actor_path = os.path.join(directory, f"{base_filename}_actor.pth")
    critic_path = os.path.join(directory, f"{base_filename}_critic.pth")

    actor_state = torch.load(actor_path, map_location=device_str, weights_only=True)
    critic_state = torch.load(critic_path, map_location=device_str, weights_only=True)

    network.actor.load_state_dict(actor_state)
    network.critic.load_state_dict(critic_state)

    network.actor_target.load_state_dict(network.actor.state_dict())
    network.critic_target.load_state_dict(network.critic.state_dict())

    if freeze_actor_layers:
        for name, p in network.actor.named_parameters():
            for pref in freeze_actor_layers:
                if name.startswith(pref):
                    p.requires_grad = False
                    print(f"[INFO] Frozen actor param: {name}")

    if freeze_critic:
        for name, p in network.critic.named_parameters():
            p.requires_grad = False
        print("[INFO] Frozen entire critic.")

    return True


def rebuild_optimizers(network, lr_actor=1e-4, lr_critic=1e-4):
    actor_params = [p for p in network.actor.parameters() if p.requires_grad]
    critic_params = [p for p in network.critic.parameters() if p.requires_grad]

    network.actor_optimizer = torch.optim.Adam(actor_params, lr=lr_actor) if len(actor_params) > 0 else None
    network.critic_optimizer = torch.optim.Adam(critic_params, lr=lr_critic) if len(critic_params) > 0 else None

    return network.actor_optimizer, network.critic_optimizer


# =========================================================
# Parallel rollout helpers
# =========================================================
def make_env(worker_id, mode="train"):
    """
    중요:
    Gazebo/ROS 이름 충돌을 피하려면 worker_id별로 launch/namespace를 분리해야 한다.
    여기서는 launch 파일명을 분리하는 예시를 둔다.
    실제로는 launch 파일이나 GazeboEnv가 worker_id / namespace를 받도록 수정하는 것이 안전하다.
    """
    if mode == "eval":
        launchfile = "multi_robot_scenario.launch"
    else:
        # 예시: worker별 launch를 다르게 쓰는 경우
        # 준비가 안 되어 있으면 일단 같은 launchfile을 쓰되,
        # 실제 실행 시 충돌이 나면 launch/namespace를 수정해야 함
        launchfile = "multi_robot_scenario.launch"

    return GazeboEnv(launchfile, environment_dim)


def maybe_load_actor_from_queue(local_actor, param_queue):
    updated = False
    latest_state = None
    while True:
        try:
            latest_state = param_queue.get_nowait()
            updated = True
        except queue.Empty:
            break
        except Exception:
            break

    if updated and latest_state is not None:
        local_actor.load_state_dict(latest_state)


def select_action_with_exploration(local_actor, state, current_expl_noise,
                                   random_near_obstacle, count_rand_actions,
                                   random_action):
    state_t = torch.Tensor(np.array(state).reshape(1, -1))
    with torch.no_grad():
        action = local_actor(state_t).cpu().numpy().flatten()

    action = (action + np.random.normal(0, current_expl_noise, size=action_dim)).clip(
        -max_action, max_action
    )

    if random_near_obstacle:
        if (
            np.random.uniform(0, 1) > 0.85
            and min(state[4:-8]) < 0.6
            and count_rand_actions < 1
        ):
            count_rand_actions = np.random.randint(8, 15)
            random_action = np.random.uniform(-1, 1, 2)

        if count_rand_actions > 0:
            count_rand_actions -= 1
            action = random_action.copy()
            action[0] = -1

    return action, count_rand_actions, random_action


def rollout_worker(worker_id, param_queue, transition_queue, stop_event,
                   initial_actor_state, worker_seed):
    set_seed(worker_seed)
    local_actor = Actor(state_dim, action_dim)
    local_actor.load_state_dict(initial_actor_state)
    local_actor.eval()

    env = make_env(worker_id, mode="train")
    time.sleep(2)

    state = env.reset()
    episode_timesteps = 0
    local_expl_noise = expl_noise
    count_rand_actions = 0
    random_action = np.zeros(2, dtype=np.float32)

    while not stop_event.is_set():
        maybe_load_actor_from_queue(local_actor, param_queue)

        if local_expl_noise > expl_min:
            local_expl_noise = local_expl_noise - ((1 - expl_min) / expl_decay_steps)
            local_expl_noise = max(local_expl_noise, expl_min)

        action, count_rand_actions, random_action = select_action_with_exploration(
            local_actor=local_actor,
            state=state,
            current_expl_noise=local_expl_noise,
            random_near_obstacle=random_near_obstacle,
            count_rand_actions=count_rand_actions,
            random_action=random_action,
        )

        a_in = [(action[0] + 1) / 2, action[1]]
        next_state, reward, done, target = env.step(a_in)

        done_bool = 0 if episode_timesteps + 1 == max_ep else int(done)
        done_flag = 1 if episode_timesteps + 1 == max_ep else int(done)

        transition_queue.put(
            {
                "worker_id": worker_id,
                "state": np.array(state, dtype=np.float32),
                "action": np.array(action, dtype=np.float32),
                "reward": float(reward),
                "done": int(done_bool),
                "next_state": np.array(next_state, dtype=np.float32),
                "episode_done": int(done_flag),
            }
        )

        state = next_state
        episode_timesteps += 1

        if done_flag:
            state = env.reset()
            episode_timesteps = 0
            count_rand_actions = 0
            random_action = np.zeros(2, dtype=np.float32)


# =========================================================
# Evaluation
# =========================================================
def evaluate(network, eval_env, epoch, eval_episodes=5):
    avg_reward = 0.0
    col = 0
    for _ in range(eval_episodes):
        state = eval_env.reset()
        done = False
        count = 0
        while not done and count < 501:
            action = network.get_action(np.array(state))
            a_in = [(action[0] + 1) / 2, action[1]]
            next_state, reward, done, target = eval_env.step(a_in)
            avg_reward += reward
            state = next_state
            count += 1
            if reward < -90:
                col += 1

    avg_reward /= eval_episodes
    avg_col = col / eval_episodes

    print("..............................................")
    print(f"Evaluation over {eval_episodes} episodes, Epoch {epoch}, Avg Reward: {avg_reward:.3f}, Collisions: {avg_col:.3f}")
    print("..............................................")
    return avg_reward


# =========================================================
# Main learner
# =========================================================
def main():
    set_seed(seed)
    mp.set_start_method("spawn", force=True)

    network = TD3(state_dim, action_dim, max_action)

    if load_pretrained:
        load_and_freeze(
            network,
            pretrained_file,
            pretrained_models,
            freeze_actor_layers=freeze_actor_layers,
            freeze_critic=freeze_critic,
            device_str=str(device),
        )
        rebuild_optimizers(
            network,
            lr_actor=actor_lr_finetune,
            lr_critic=critic_lr_finetune,
        )

    if load_model:
        try:
            network.load(file_name, models_dir)
            print(f"[INFO] Loaded existing finetune model: {file_name}")
        except Exception as e:
            print(f"[WARN] Could not load finetune model. Start from current weights. {e}")

    replay_buffer = ReplayBuffer(buffer_size, seed)
    eval_env = make_env(worker_id=-1, mode="eval")
    time.sleep(3)

    evaluations = []
    best_eval = -float("inf")

    transition_queue = mp.Queue(maxsize=50000)
    param_queues = [mp.Queue(maxsize=4) for _ in range(num_workers)]
    stop_event = mp.Event()

    initial_actor_state = copy.deepcopy(network.actor.state_dict())

    workers = []
    for wid in range(num_workers):
        p = mp.Process(
            target=rollout_worker,
            args=(
                wid,
                param_queues[wid],
                transition_queue,
                stop_event,
                initial_actor_state,
                seed + wid + 1,
            ),
            daemon=True,
        )
        p.start()
        workers.append(p)

    print(f"[INFO] Started {num_workers} rollout workers.")

    global_timestep = 0
    update_count = 0
    epoch = 1
    last_eval_timestep = 0

    try:
        while global_timestep < max_timesteps:
            collected = 0

            while collected < 1000 and global_timestep < max_timesteps:
                try:
                    item = transition_queue.get(timeout=queue_timeout)
                except queue.Empty:
                    break

                replay_buffer.add(
                    item["state"],
                    item["action"],
                    item["reward"],
                    item["done"],
                    item["next_state"],
                )
                global_timestep += 1
                collected += 1

            if replay_buffer.size() >= max(batch_size, warmup_steps):
                learner_updates = max(1, collected * train_steps_per_env_step)
                network.train(
                    replay_buffer,
                    iterations=learner_updates,
                    batch_size=batch_size,
                    discount=discount,
                    tau=tau,
                    policy_noise=policy_noise,
                    noise_clip=noise_clip,
                    policy_freq=policy_freq,
                )
                update_count += learner_updates

            if update_count > 0 and update_count % sync_interval == 0:
                actor_state_cpu = {
                    k: v.detach().cpu()
                    for k, v in network.actor.state_dict().items()
                }
                for pq in param_queues:
                    try:
                        while True:
                            pq.get_nowait()
                    except Exception:
                        pass
                    pq.put(actor_state_cpu)

            if global_timestep - last_eval_timestep >= eval_freq:
                last_eval_timestep = global_timestep
                eval_reward = evaluate(network, eval_env, epoch, eval_episodes=eval_ep)
                evaluations.append(eval_reward)

                if save_model:
                    network.save(f"{file_name}_latest", models_dir)
                    if eval_reward > best_eval:
                        best_eval = eval_reward
                        network.save(f"{file_name}_best", models_dir)
                        print(f"[INFO] Saved new best model. best_eval={best_eval:.3f}")

                np.save(os.path.join(results_dir, file_name), evaluations)
                epoch += 1

                print(
                    f"[INFO] timestep={global_timestep}, replay={replay_buffer.size()}, "
                    f"updates={update_count}, best_eval={best_eval:.3f}"
                )

        final_eval = evaluate(network, eval_env, epoch, eval_episodes=eval_ep)
        evaluations.append(final_eval)

        if save_model:
            network.save(f"{file_name}_latest", models_dir)
            if final_eval > best_eval:
                network.save(f"{file_name}_best", models_dir)

        np.save(os.path.join(results_dir, file_name), evaluations)

    finally:
        stop_event.set()
        time.sleep(2)
        for p in workers:
            if p.is_alive():
                p.terminate()
        for p in workers:
            p.join(timeout=2)
        print("[INFO] All workers terminated.")


if __name__ == "__main__":
    main()