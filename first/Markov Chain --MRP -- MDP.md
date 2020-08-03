# Markov Chain --MRP -- MDP

为什么要提到马尔可夫决策过程？

 Markov Decision Process can model a lot of real-world problem. It formally describes the framework of reinforcement learning 

## 马尔可夫过程

下一时刻的状态只与当前时刻的状态有关

p(st+1|st) =p(st+1|ht)

## 马尔可夫奖励过程

在MP基础上加入Reward和折扣因子γ

**为什么加入γ？**

1. 在无限过程中总奖励才能收敛
2. 立即奖励总是要比将来的奖励更重要

Gt:从t时刻直到过程结束的折扣奖励总和

Vt（s）: Gt在状态s时Gt的期望

![image-20200802173637903](https://i.loli.net/2020/08/02/Vc7gHp1nsywzajP.png)

解贝尔曼方程：

1. 矩阵求解

![image-20200802173857175](https://i.loli.net/2020/08/02/IJlTc7pzfgOovrY.png)
$$
V = R + \gamma PV
$$

$$
V = (1-\gamma P)^{-1} V
$$

2. 迭代方法求解

   - 动态规划
   - 蒙特卡洛

   ![image-20200802203332391](https://i.loli.net/2020/08/02/IX2YtNJMZ6PvdDf.png)

   必须要等到过程结束才能进行计算

   - T-D

# MDP马尔可夫决策过程

相比与MRP，MDP加入了策略或者说动作，因此也更加复杂，以下两张图说明了两者的区别

![image-20200802203916741](https://i.loli.net/2020/08/02/4IgA6D1sLaBORGP.png)

![image-20200802204237553](https://i.loli.net/2020/08/02/iN2JzK57pfPA6Qr.png)

MDP (S,A,P,R,γ)+policy,以下为价值函数的递推公式：

![image-20200802210002716](https://i.loli.net/2020/08/02/vpgOoTxL3t8UMu1.png)

![image-20200802210041261](https://i.loli.net/2020/08/02/yft5hg47p81Kj3k.png)

## 求解

- 动态规划/policy evaluation

$$
v^{\pi}(s)=\sum_{a \in A} \pi(a \mid s)\left(R(s, a)+\gamma \sum_{s^{\prime} \in S} P\left(s^{\prime} \mid s, a\right) v^{\pi}\left(s^{\prime}\right)\right)
$$

通过不断迭代可得到各个状态的稳定解。

**优点**：可解释性强

**缺点**：只能解决离散且规模小的问题；必须知道环境模型（条件转移概率）

- MDP control

  - 策略迭代

  ![image-20200803140913144](https://i.loli.net/2020/08/03/clKhgWp9zn1YaFy.png)

  分两个过程：先进行策略评估，再进行策略改进

  - 值迭代

  ![image-20200803162557833](https://i.loli.net/2020/08/03/jpKWJD1ZMAktdSm.png)

$$
v_{*}(s) \leftarrow \max _{a \in \mathcal{A}} \mathcal{R}_{s}^{a}+\gamma \sum_{s^{\prime} \in \mathcal{S}} \mathcal{P}_{s s^{\prime}}^{a} v_{*}\left(s^{\prime}\right)
$$

​	根据贝尔曼最优方程依次迭代，直到收敛得到最优值函数的稳定解，根据最优值函数直接得到最优策略。

![image-20200803163404831](C:%5CUsers%5C86156%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20200803163404831.png)

**提问**：理论的证明会了吗