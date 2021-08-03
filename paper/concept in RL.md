# concept in RL



**RL** is the study of agents and how they learn by trial and error. rewarding or punishing an agent for its behavior makes it more likely to repeat or forego that behavior in the future.

![../_images/rl_diagram_transparent_bg.png](https://spinningup.openai.com/en/latest/_images/rl_diagram_transparent_bg.png)

**Environment** 是agent所处的世界，**agent** 是机器人之类的智能体，它通过与环境交互：作出动作从而改变其所处的环境，同时观察环境（环境的一部分，它无法观察到全部的世界），并作出下一步的动作与环境交互...

**reward**用来评价当前的**world state**， agent的目标就是要通过学习一定的行为动作，获得最大的累积reward

## terminology

- states and observations ：状态是可以完全描述世界的状态，观察是智能体观察到世界的状态，智能体作出决策动作的依据也是观察

- action spaces ：离散的和连续的

- policies ：确定性策略 和 随机策略，随机策略又有分类型策略和对角高斯策略。

  分类型策略输入观察，输出离散动作的概率值。

- trajectories ：状态和行为序列 
  $$
  \begin{align*}
  \tau = (S_0, a_0, S_1, a_1, ...)
  \end{align*}
  $$
  确定型的服从某一函数关系，随机型的服从某一概率分布

  ![s_{t+1} = f(s_t, a_t)](https://spinningup.openai.com/en/latest/_images/math/16da6346104894fb6a673473cbfc9ffeba6471fa.svg)

  ![s_{t+1} \sim P(\cdot|s_t, a_t).](https://spinningup.openai.com/en/latest/_images/math/872390af4f5b2541d17e7ef2bfaecbe1e9746d94.svg)

- different formulations of return : 
  $$
  r_t =  R(s_t,a_t,s_{t+1})
  $$
  **finite-horizon undiscounted return**![R(\tau) = \sum_{t=0}^T r_t.](https://spinningup.openai.com/en/latest/_images/math/b2466507811fc9b9cbe2a0a51fd36034e16f2780.svg)

  **infinite-horizon discounted return**![R(\tau) = \sum_{t=0}^{\infty} \gamma^t r_t.](https://spinningup.openai.com/en/latest/_images/math/bf49428c66c91a45d7b66a432450ee49a3622348.svg)

- the RL optimization problem

  the goal in RL is to select a policy which maximizes **expected return** when the agent acts according to it.

  

  ![2020-07-16 14-08-33 的屏幕截图](/home/chenmao/Pictures/2020-07-16 14-08-33 的屏幕截图.png)

  

  

-  value functions

值函数![V^{\pi}(s) = \underE{\tau \sim \pi}{R(\tau)\left| s_0 = s\right.}](https://spinningup.openai.com/en/latest/_images/math/e043709b46c9aa6811953dabd82461db6308fe19.svg)

动作值函数![Q^{\pi}(s,a) = \underE{\tau \sim \pi}{R(\tau)\left| s_0 = s, a_0 = a\right.}](https://spinningup.openai.com/en/latest/_images/math/85d41c8c383a96e1ed34fc66f14abd61b132dd28.svg)

最优值函数![V^*(s) = \max_{\pi} \underE{\tau \sim \pi}{R(\tau)\left| s_0 = s\right.}](https://spinningup.openai.com/en/latest/_images/math/01d48ea453ecb7b560ea7d42144ae24422fbd0eb.svg)

最优动作值函数![Q^*(s,a) = \max_{\pi} \underE{\tau \sim \pi}{R(\tau)\left| s_0 = s, a_0 = a\right.}](https://spinningup.openai.com/en/latest/_images/math/bc92e8ce1cf0aaa212e144d5ed74e3b115453cb6.svg)

### Bellman Equations

![\begin{align*} V^{\pi}(s) &= \underE{a \sim \pi \\ s'\sim P}{r(s,a) + \gamma V^{\pi}(s')}, \\ Q^{\pi}(s,a) &= \underE{s'\sim P}{r(s,a) + \gamma \underE{a'\sim \pi}{Q^{\pi}(s',a')}}, \end{align*}](https://spinningup.openai.com/en/latest/_images/math/7e4a2964e190104a669406ca5e1e320a5da8bae0.svg)

The Bellman equations for the optimal value functions are![\begin{align*} V^*(s) &= \max_a \underE{s'\sim P}{r(s,a) + \gamma V^*(s')}, \\ Q^*(s,a) &= \underE{s'\sim P}{r(s,a) + \gamma \max_{a'} Q^*(s',a')}. \end{align*}](https://spinningup.openai.com/en/latest/_images/math/f8ab9b211bc9bb91cde189360051e3bd1f896afa.svg)

### Advantage Functions

评价一个动作相对于其他动作有多好

![A^{\pi}(s,a) = Q^{\pi}(s,a) - V^{\pi}(s).](https://spinningup.openai.com/en/latest/_images/math/3748974cc061fb4065fa46dd6271395d59f22040.svg)