# Tao

[<img src="https://img.shields.io/badge/license-MIT-blue">](https://github.com/amulil/tao)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![docs](https://img.shields.io/github/deployments/vwxyzjn/cleanrl/Production?label=docs&logo=vercel)]()

## 算法实现

- [x] PPO
- [ ] DDPG
- [ ] SAC
- [ ] DQN
- [ ] TD3

## 基准

- [ ] sb3
- [ ] cleanrl

## 使用

```python
# ppo example
# train model
model = PPO(env_id="CartPole-v1")
model.learn()

# save model
import torch
is_save = True
if is_save:
    torch.save(agent.state_dict(), "./ppo.pt")
    
# load model
model = PPO(env_id="CartPole-v1")
model.load_state_dict(torch.load("./ppo.pt", map_location="cpu"))
model.eval()
```