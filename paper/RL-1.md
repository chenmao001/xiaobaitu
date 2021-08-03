## Reward

- 反馈信号，用来衡量t时刻的action有好
- RL的目标就是最大化reward

## Sequential Decision Making

- 最大化奖励
- action有long term的影响
- 奖励有延迟
- 权衡立即奖励和长期奖励

## Major Components of an RL Agent

- policy:agent的动作函数
  - 从状态到动作的映射

![](https://i.loli.net/2020/08/02/oDYOcRKzvfiyUdF.png)

- value function:评价状态或动作有多好
  - 未来reward的**折现**和的**期望**
  - v and Q
- model：A model predicts what the environment will do next

![image-20200802110247950](https://i.loli.net/2020/08/02/oPztaLQGXS4cgKV.png)

# MDP求解RL

基本术语、概念

agent分类：

- 从agent学的是什么来分：value-based(直接学习值函数，隐式的学习策略),policy-based（直接学习策略）,actor-critic（前两者结合）

![image-20200802133841364](https://i.loli.net/2020/08/02/R6cBQypAU2F4NLZ.png)

![image-20200802133746937](https://i.loli.net/2020/08/02/DUJIBfyHsAG9QoK.png)

- model-based(可能没有学习策略和价值函数的过程),model-free

![image-20200802112106777](https://i.loli.net/2020/08/02/emvJBqncdEjXayW.png)

Two Fundamental Problems in Sequential Decision Making

1. Planning

Given model about how the environment works.

Compute how to act to maximize expected reward without external interaction.

2. Reinforcement learning

 Agent doesn’t know how world works

 Interacts with world to implicitly learn how world works

 Agent improves policy (also involves planning)

## 探索与开发

