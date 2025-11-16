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
4. Prefarably the network controls the robot directly and doesnt need kinematics to execute the task

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
    * Doesnt need reward shaping, can work with sparse rewards
    * Sim to Real experiments also in the paper

3. Learning Synergies between Pushing and Grasping with Self-supervised Deep Reinforcement Learning
    * Learn two networks, one for pushing and one for grasping which take in camera images and output the expected utility of pushing/ grasping with each motion primitive
    * Concerns:
        * I dont really care about pushing right now
        * After you find the location, it seems like the actual execution of trajectory is still done by kinematics
        * Needs depth camera, performance much worse with just RGB camera
     

### Chosen algorithm:
* DDPG with Hindsight Experience Replay
  * Task: 
  * Action space:
  * Observation Space
  * Reward Function
