# policy

## on-policy 

**on-policy**: that is, they don’t use old data, **disadvantage**: which makes them weaker on sample efficiency. **advantage:** these algorithms directly optimize the objective you care about

- Vanilla Policy Gradient(VPG)

- TRPO 

- PPO

## off-policy

**off-policy: **they are able to reuse old data very efficiently; **disadvantage:** there are no guarantees that doing a good job of satisfying Bellman’s equations leads to having great policy performance. 

- DDPG

- DQN

- TD3

- SAC



