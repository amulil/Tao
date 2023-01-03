import random
import time

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


class DQN(nn.Module):
    def __init__(
        self,
        model_name="DQN",
        track=False,
        wandb_project_name="tao",
        wandb_entity=None,
        env_id=None,
        capture_video=False,
        total_timesteps=500000,  # atari: 10000000
        learning_rate=2.5e-4,  # atari: 1e-4
        buffer_size=10000,  # atari: 1000000
        gamma=0.99,
        target_network_frequency=500,  # 控制 NN 多久更新一次 atari: 1000
        batch_size=128,  # atari: 32
        start_e=1,
        end_e=0.05,  # atari: 0.01
        exploration_fraction=0.5,  # atari: 0.1
        learning_starts=10000,  # atari: 80000
        train_frequency=10,  # atari: 4
        seed=1,
        device="cpu",
        atari_env=False,
        torch_deterministic=True,
    ):
        super().__init__()
        self.model_name = model_name
        self.track = track
        self.wandb_project_name = wandb_project_name
        self.wandb_entity = wandb_entity
        self.env_id = env_id
        self.capture_video = capture_video
        self.total_timesteps = total_timesteps
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.target_network_frequency = target_network_frequency
        self.batch_size = batch_size
        self.start_e = start_e
        self.end_e = end_e
        self.exploration_fraction = exploration_fraction
        self.learning_starts = learning_starts
        self.train_frequency = train_frequency
        self.seed = seed
        self.run_name = f"{self.env_id}__{self.model_name}__{self.seed}__{total_timesteps}"
        self.device = device
        self.torch_deterministic = torch_deterministic
        self.atari_env = atari_env
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = self.torch_deterministic
        self.envs = gym.vector.SyncVectorEnv([self._make_env(self.env_id, self.seed, 0, self.capture_video, self.run_name)])
        if not self.atari_env:
            self.q_network = nn.Sequential(
                layer_init(nn.Linear(np.array(self.envs.single_observation_space.shape).prod(), 120)),
                nn.ReLU(),
                layer_init(nn.Linear(120, 84)),
                nn.ReLU(),
                layer_init(nn.Linear(84, self.envs.single_action_space.n)),
            )
            self.target_network = nn.Sequential(
                layer_init(nn.Linear(np.array(self.envs.single_observation_space.shape).prod(), 120)),
                nn.ReLU(),
                layer_init(nn.Linear(120, 84)),
                nn.ReLU(),
                layer_init(nn.Linear(84, self.envs.single_action_space.n)),
            )

        if self.atari_env:
            self.q_network = nn.Sequential(
                layer_init(nn.Conv2d(4, 32, 8, stride=4)),
                nn.ReLU(),
                layer_init(nn.Conv2d(32, 64, 4, stride=2)),
                nn.ReLU(),
                layer_init(nn.Conv2d(64, 64, 3, stride=1)),
                nn.ReLU(),
                nn.Flatten(),
                layer_init(nn.Linear(64 * 7 * 7, 512)),
                nn.ReLU(),
                layer_init(nn.Linear(512, self.envs.single_action_space.n)),
            )
            self.target_network = nn.Sequential(
                layer_init(nn.Conv2d(4, 32, 8, stride=4)),
                nn.ReLU(),
                layer_init(nn.Conv2d(32, 64, 4, stride=2)),
                nn.ReLU(),
                layer_init(nn.Conv2d(64, 64, 3, stride=1)),
                nn.ReLU(),
                nn.Flatten(),
                layer_init(nn.Linear(64 * 7 * 7, 512)),
                nn.ReLU(),
                layer_init(nn.Linear(512, self.envs.single_action_space.n)),
            )

    def _nn(self, x):
        return self.network(x)

    def _make_env(self, env_id, seed, idx, capture_video, run_name):
        def thunk_base():
            env = gym.make(env_id)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            if capture_video:
                if idx == 0:
                    env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
            env.seed(seed)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
            return env

        def thunk_atari():
            env = gym.make(env_id)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            if capture_video:
                if idx == 0:
                    env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
            env = NoopResetEnv(env, noop_max=30)
            env = MaxAndSkipEnv(env, skip=4)
            env = EpisodicLifeEnv(env)
            if "FIRE" in env.unwrapped.get_action_meanings():
                env = FireResetEnv(env)
            env = ClipRewardEnv(env)
            env = gym.wrappers.ResizeObservation(env, (84, 84))
            env = gym.wrappers.GrayScaleObservation(env)
            env = gym.wrappers.FrameStack(env, 4)
            env.seed(seed)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
            return env

        if self.atari_env:
            return thunk_atari

        return thunk_base

    def learn(self):
        if self.track:
            import wandb

            wandb.init(
                project=self.wandb_project_name,
                entity=self.wandb_entity,
                sync_tensorboard=True,
                # config=vars(self),
                name=self.run_name,
                monitor_gym=True,
                save_code=True,
            )
        writer = SummaryWriter(f"runs/{self.run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(self).items()])),
        )

        self.seed
        device = self.device
        envs = self.envs

        agent = self.to(device)
        optimizer = optim.Adam(agent.q_network.parameters(), lr=self.learning_rate)
        self.target_network.load_state_dict(agent.q_network.state_dict())

        rb = ReplayBuffer(
            self.buffer_size,
            self.envs.single_observation_space,
            self.envs.single_action_space,
            self.device,
            handle_timeout_termination=True,
        )

        start_time = time.time()
        # TRY NOT TO MODIFY: start the game
        obs = envs.reset()
        for global_step in range(self.total_timesteps):
            # ALGO LOGIC: put action logic here
            epsilon = linear_schedule(self.start_e, self.end_e, self.exploration_fraction * self.total_timesteps, global_step)
            if random.random() < epsilon:
                actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
            else:
                if self.atari_env:
                    obs = obs / 255.0
                q_values = self.q_network(torch.Tensor(obs).to(device))
                actions = torch.argmax(q_values, dim=1).cpu().numpy()

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, rewards, dones, infos = envs.step(actions)

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            for info in infos:
                if "episode" in info.keys():
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                    writer.add_scalar("charts/epsilon", epsilon, global_step)
                    break

            # TRY NOT TO MODIFY: save data to reply buffer; handle `terminal_observation`
            real_next_obs = next_obs.copy()
            for idx, d in enumerate(dones):
                if d:
                    real_next_obs[idx] = infos[idx]["terminal_observation"]
            rb.add(obs, real_next_obs, actions, rewards, dones, infos)

            # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
            obs = next_obs

            # ALGO LOGIC: training.
            if global_step > self.learning_starts:
                if global_step % self.train_frequency == 0:
                    data = rb.sample(self.batch_size)
                    if self.atari_env:
                        data.next_observations = data.next_observations / 255.0
                        data.observations = data.observations / 255.0
                    with torch.no_grad():
                        target_max, _ = self.target_network(data.next_observations).max(dim=1)
                        td_target = data.rewards.flatten() + self.gamma * target_max * (1 - data.dones.flatten())
                    old_val = self.q_network(data.observations).gather(1, data.actions).squeeze()
                    loss = F.mse_loss(td_target, old_val)

                    if global_step % 100 == 0:
                        writer.add_scalar("losses/td_loss", loss, global_step)
                        writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                        print("SPS:", int(global_step / (time.time() - start_time)))
                        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

                    # optimize the model
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # update the target network
                if global_step % self.target_network_frequency == 0:
                    self.target_network.load_state_dict(self.q_network.state_dict())
