import os

from tao import utils
from tao.dqn.dqn import DQN
from tao.ppo.ppo import PPO

# Read version from file
version_file = os.path.join(os.path.dirname(__file__), "version.txt")
with open(version_file) as file_handler:
    __version__ = file_handler.read().strip()

__all__ = ["PPO", "DQN", "utils"]
