import numpy as np
import torch
import torch.nn as nn


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


class DDPG(nn.Module):
    def __init__(
        self,
        model_name="DDPG",
        track=False,
        wandb_project_name="tao",
        wandb_entity=None,
        env_id=None,
        capture_video=False,
        total_timesteps=500000,  # atari: 1000000
        learning_rate=2.5e-4,
        seed=1,
        device="cpu",
        atari_env=False,
        torch_deterministic=True,
    ):
        super().__init__()
