import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from velodyne_env import GazeboEnv


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


class TD3(object):
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim).to(device)

    def get_action(self, state):
        state = torch.Tensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def load(self, filename, directory):
        self.actor.load_state_dict(
            torch.load("%s/%s_actor.pth" % (directory, filename), map_location=device)
        )


# =========================
# Parameters
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 0
max_ep = 500
file_name = "TD3_velodyne"

num_eval_episodes = 10000   # 평가 episode 수
environment_dim = 20
robot_dim = 4
state_dim = environment_dim + robot_dim
action_dim = 2

torch.manual_seed(seed)
np.random.seed(seed)

# =========================
# Environment / Network
# =========================
env = GazeboEnv("multi_robot_scenario.launch", environment_dim)
time.sleep(5)

network = TD3(state_dim, action_dim)
try:
    network.load(file_name, "./pytorch_models")
except Exception as e:
    raise ValueError(f"Could not load the stored model parameters: {e}")


# =========================
# Evaluation
# =========================
success_count = 0

for episode in range(num_eval_episodes):
    state = env.reset()
    done = False
    episode_timesteps = 0
    episode_success = False

    while not done and episode_timesteps < max_ep:
        action = network.get_action(np.array(state))

        # linear velocity: [0, 1], angular velocity: [-1, 1]
        a_in = [(action[0] + 1) / 2, action[1]]

        next_state, reward, done, target = env.step(a_in)

        # target이 True면 성공으로 간주
        if target:
            episode_success = True

        state = next_state
        episode_timesteps += 1

        # max step 도달 시 episode 종료 처리
        if episode_timesteps >= max_ep:
            done = True

    if episode_success:
        success_count += 1

    # 중간 진행상황 출력
    if (episode + 1) % 100 == 0:
        success_rate = 100.0 * success_count / (episode + 1)
        print(
            f"[Evaluation] Episode {episode + 1}/{num_eval_episodes} | "
            f"Successes: {success_count} | Success Rate: {success_rate:.2f}%"
        )

# =========================
# Final result
# =========================
final_success_rate = 100.0 * success_count / num_eval_episodes
print("\n========== Evaluation Finished ==========")
print(f"Total Episodes   : {num_eval_episodes}")
print(f"Successful Epis. : {success_count}")
print(f"Success Rate     : {final_success_rate:.2f}%")
print("=========================================\n")