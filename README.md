# lerobot_drl
Deep Reinforcement Learning Experiments with lerobot

### Main questions to answer
1. Can we train lerobot to pick up and move objects without any expert trajectories?
2. How well do policies transfer from sim to real?
3. Can we use imitation learning to initialise a policy and then use reinforcement learning to generalise?

### Constraints:
1. RGB cameras only for now
2. Lerobot
3. Task should be reproducible in sim

### Options:
1. Deep Spatial Autoencoders for Visuomotor Learning
   * Use an autoencoder to learn to extract features from images unsupervised and then a RL algorithm to learn dynamic motion
   * Concerns:
       * RL algorithm looks quite convoluted
       * Implementation looks complex of the autoencoder and the RL algorithm
       * No code available as reference

2. Hindsight Experience Replay
  * Can be used with any off policy RL algorithm like DQN and DDPG and helps make RL faster by relabelling goals
  * Concerns:
    * DQN needs a discrete action space so not sure how that extends to lerobot
  * Lots of example implementations available online
