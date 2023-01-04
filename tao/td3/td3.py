import random
import time

import gym
import numpy as np
import pybullet_envs  # noqa
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


class TD3(nn.Module):
    def __init__(
        self,
        model_name="TD3",
        track=False,
        wandb_project_name="tao",
        wandb_entity=None,
        env_id=None,
        capture_video=False,
        total_timesteps=int(1e6),
        learning_rate=3e-4,
        buffer_size=int(1e6),
        gamma=0.99,
        tau=0.005,
        batch_size=256,
        exploration_noise=0.1,
        learning_starts=25000,
        policy_frequency=2,
        noise_clip=0.2,
        seed=1,
        device="cpu",
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
        self.tau = tau
        self.batch_size = batch_size
        self.exploration_noise = exploration_noise
        self.learning_starts = learning_starts
        self.policy_frequency = policy_frequency
        self.noise_clip = noise_clip
        self.seed = seed
        self.run_name = f"{self.env_id}__{self.model_name}__{self.seed}__{total_timesteps}"
        self.device = device
        self.torch_deterministic = torch_deterministic
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = self.torch_deterministic
        self.envs = gym.vector.SyncVectorEnv([self._make_env(self.env_id, self.seed, 0, self.capture_video, self.run_name)])

        self.q1 = nn.Sequential(
            layer_init(
                nn.Linear(
                    np.array(self.envs.single_observation_space.shape).prod() + np.prod(self.envs.single_action_space.shape),
                    256,
                )
            ),
            nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 1)),
        )
        self.q2 = nn.Sequential(
            layer_init(
                nn.Linear(
                    np.array(self.envs.single_observation_space.shape).prod() + np.prod(self.envs.single_action_space.shape),
                    256,
                )
            ),
            nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 1)),
        )
        self.q1_target = nn.Sequential(
            layer_init(
                nn.Linear(
                    np.array(self.envs.single_observation_space.shape).prod() + np.prod(self.envs.single_action_space.shape),
                    256,
                )
            ),
            nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 1)),
        )
        self.q2_target = nn.Sequential(
            layer_init(
                nn.Linear(
                    np.array(self.envs.single_observation_space.shape).prod() + np.prod(self.envs.single_action_space.shape),
                    256,
                )
            ),
            nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 1)),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(self.envs.single_observation_space.shape).prod(), 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, np.prod(self.envs.single_action_space.shape))),
            nn.Tanh(),
        )
        self.actor_target = nn.Sequential(
            layer_init(nn.Linear(np.array(self.envs.single_observation_space.shape).prod(), 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, np.prod(self.envs.single_action_space.shape))),
            nn.Tanh(),
        )

    def _scale_action_value(self, action_value):
        return self.envs.action_space.high * action_value

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

        self.to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.q1_target.load_state_dict(self.q1.state_dict())
        q_optimizer = optim.Adam(list(self.q1.parameters()), lr=self.learning_rate)
        actor_optimizer = optim.Adam(list(self.actor.parameters()), lr=self.learning_rate)

        envs.single_observation_space.dtype = np.float32
        rb = ReplayBuffer(
            self.buffer_size,
            envs.single_observation_space,
            envs.single_action_space,
            device,
            handle_timeout_termination=True,
        )
        start_time = time.time()

        # TRY NOT TO MODIFY: start the game
        obs = envs.reset()
        for global_step in range(self.total_timesteps):
            # ALGO LOGIC: put action logic here
            if global_step < self.learning_starts:
                actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
            else:
                with torch.no_grad():
                    actions = self._scale_action_value(self.actor(torch.Tensor(obs))).to(device)
                    actions += torch.normal(0, self.exploration_noise)
                    actions = actions.cpu().numpy().clip(envs.single_action_space.low, envs.single_action_space.high)

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, rewards, dones, infos = envs.step(actions)

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            for info in infos:
                if "episode" in info.keys():
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
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
                data = rb.sample(self.batch_size)
                with torch.no_grad():
                    # rand_like 返回一个和输入大小相同的张量，其由均值为0、方差为1的标准正态分布填充
                    clipped_noise = (torch.randn_like(data.actions, device=self.device)).clamp(
                        -self.noise_clip, self.noise_clip
                    ) * self.envs.single_action_space.high[0]
                    next_state_actions = (self.actor_target(data.next_observations) + clipped_noise).clamp(
                        self.envs.single_action_space.low[0], self.envs.single_action_space.high[0]
                    )
                    qf1_next_target = self.q1_target(data.next_observations, next_state_actions)
                    qf2_next_target = self.q2_target(data.next_observations, next_state_actions)
                    min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
                    next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * self.gamma * (
                        min_qf_next_target
                    ).view(-1)

                qf1_a_values = self.q1(data.observations, data.actions).view(-1)
                qf2_a_values = self.q2(data.observations, data.actions).view(-1)
                qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
                qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
                qf_loss = qf1_loss + qf2_loss

                # optimize the model
                q_optimizer.zero_grad()
                qf_loss.backward()
                q_optimizer.step()

                # update the target network (soft-update q param not delayed)
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                if global_step % self.policy_frequency == 0:
                    actor_loss1 = -self.q1(data.observations, self.actor(data.observations)).mean()
                    actor_loss2 = -self.q2(data.observations, self.actor(data.observations)).mean()
                    actor_loss = actor_loss1 + actor_loss2
                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    for param, target_param in zip(self.q2.parameters(), self.q2_target.parameters()):
                        target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                if global_step % 100 == 0:
                    writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                    writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                    writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                    writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                    writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                    writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                    print("SPS:", int(global_step / (time.time() - start_time)))
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
