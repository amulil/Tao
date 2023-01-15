# Tao

[<img src="https://img.shields.io/badge/license-MIT-blue">](https://github.com/amulil/tao)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![docs](https://img.shields.io/github/deployments/vwxyzjn/cleanrl/Production?label=docs&logo=vercel)]()
## 算法原理

- SADRL
0. [深度强化学习（DRL）算法汇总](https://zhuanlan.zhihu.com/p/595383059)
1. [深度强化学习（DRL）算法 1 —— REINFORCE](https://zhuanlan.zhihu.com/p/574479257)
2. [深度强化学习（DRL）算法 2 —— PPO 之 Clipped Surrogate Objective 篇](https://zhuanlan.zhihu.com/p/574810519)
3. [深度强化学习（DRL）算法 2 —— PPO 之 GAE 篇](https://zhuanlan.zhihu.com/p/577598804)
4. [深度强化学习（DRL）算法 3 —— Deep Q-learning(DQN)](https://zhuanlan.zhihu.com/p/595728811)
5. [深度强化学习（DRL）算法 4 —— Deep Deterministic Policy Gradient (DDPG)](https://zhuanlan.zhihu.com/p/595568507)
6. [深度强化学习（DRL）算法 5 —— Twin Delayed Deep Deterministic Policy Gradient (TD3)](https://zhuanlan.zhihu.com/p/596086108)
7. [深度强化学习（DRL）算法 6 —— Soft Actor-Critic (SAC)]()
8. [深度强化学习（DRL）算法 附录 1 —— 贝尔曼公式](https://zhuanlan.zhihu.com/p/581273520)
9. [深度强化学习（DRL）算法 附录 2 —— 策略迭代和价值迭代](https://zhuanlan.zhihu.com/p/595732361)
10. [深度强化学习（DRL）算法 附录 3 —— 蒙特卡洛方法（MC）和时序差分（TD）](https://zhuanlan.zhihu.com/p/595786697)
11. [深度强化学习（DRL）算法 附录 4 —— 一些常用概念（KL 散度、最大熵 MDP etc.）](https://zhuanlan.zhihu.com/p/596733815)
12. [深度强化学习（DRL）算法 附录 5 —— CV 基础回顾篇](https://zhuanlan.zhihu.com/p/597512439)
13. [深度强化学习（DRL）算法 附录 6 —— NLP 回顾之基础模型篇](https://zhuanlan.zhihu.com/p/597512525)
14. [深度强化学习（DRL）算法 附录 6 —— NLP 回顾之预训练模型篇](https://zhuanlan.zhihu.com/p/599095755)

- MADRL

## 算法实现
- 单智能体
- [x] PPO
  - [x] discrete action
  - [x] continuous action
  - [x] atari
- [x] DDPG
- [ ] SAC
- [x] DQN
  - [x] discrete action
  - [x] atari
- [x] TD3
- 多智能体

- [ ] MAPPO(IPPO) 
- [ ] HATRPO/HAPPO
- [ ] MA Transformer

## 基准

- [x] tao
- [ ] cleanrl
- [ ] sb3
- [ ] openai/baselines

### [Tao baseline](https://wandb.ai/amulil/tao/reports/Tao-baseline--VmlldzozMjU0OTQ3)
-------
#### Cartpole-v1
![](./docs/static/cartpole.png)
#### atari/BreakoutNoFrameskip-v4
![](./docs/static/breakout.png)

## 本地运行
```bash
git clone https://github.com/amulil/Tao.git && cd tao
poetry install
poetry run jupyter notebook # run examples in notebook
```

## 使用

```python
# train model
from tao import PPO
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