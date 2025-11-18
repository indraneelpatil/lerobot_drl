# lerobot_drl
Deep Reinforcement Learning Experiments with lerobot

### Main questions to answer
1. Can we train lerobot to pick up and move objects without any expert trajectories?
2. How well do policies transfer from sim to real?
3. Can we use imitation learning to initialise a policy and then use reinforcement learning to generalise?
4. How does RL extend to fine tuning a VLA model?

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
    * All training of DDPG is done in simulation
    * Sim to Real experiments also in the paper
    * DDPG with HER
      * Box positions are obtained using a separately trained CNN
      * Action space: Relative gripper position at next timestep and gripper position (4d)
      * Observation Space: Joint positions and velocities of the robot, positions and rotations of all objects, all object positions are relative to the gripper position
      * Goals: Desired positions of objects
      * Hindsight experience replay needs the policies to be goal conditioned, so if the goal definition for objects is in terms of positions the observation should be position too
      * I want something which goes from images -> joint space,maybe just DDPG with normal experience replay?

3. Learning Synergies between Pushing and Grasping with Self-supervised Deep Reinforcement Learning
    * Learn two networks, one for pushing and one for grasping which take in camera images and output the expected utility of pushing/ grasping with each motion primitive
    * Concerns:
        * I dont really care about pushing right now
        * After you find the location, it seems like the actual execution of trajectory is still done by kinematics
        * Needs depth camera, performance much worse with just RGB camera
4. CONTINUOUS CONTROL WITH DEEP REINFORCEMENT LEARNING
    * Actor-Critic model free algorithm that can operate in continuous action spaces based on DDPG
    * Can learn policies end to end
    * Concerns:
        * All experiments in paper are on low dimensional robot, no experiments on high DOF manipulators, may work may not work
5. Precise and Dexterous Robotic Manipulation via Human-in-the-Loop Reinforcement Learning
   * Called HIL-SERL (Human in the Loop Sample Efficient Robotic Reinforcement Learning)
   * 1 to 2.5 hours of training time and out performs IL baselines
   * Use a binary classifier as a sparse reward signal
   * Collect some demonstrations for the demo buffer
   * Train using RL and human interventions
     * Use a pretrained visual backbone for policy learning
     * Off policy RL algorithm that incorporates human demonstrations and corrections
   * https://huggingface.co/docs/lerobot/en/hilserl
   * Reward function: Trained Binary Classifier
   * RL Algorithm: RLPD (Off policy)
   * Observation space: Images from wirst mounted and side cameras, end affector poses, twists, forces/ torques and current gripper status of both arms
   * Action Space: Direct feedforward wrenches in end affector frame (desired accelerations)
   * Action Space 2: 6D Cartesian twist target for downstream impedance controller + discrete control for gripper
   * ResNet-10 model pretrained on ImageNet
   * Ego centric formulation: Action output from policy is relative to current end affector frame
   * Gripper control is treated as a separate MDP and controlled by separate network
   * Wrist cameras are useful for providing spatial generalisation of the policy due to the ego centric views they provide
   * But wrist cameras alone dont provide a full view of the environment
     

### Chosen algorithm:
* TODO
  * Task: 
  * Action space: Relative gripper position at next timestep and gripper position (4d)
  * Observation Space: Joint positions and velocities of the robot, positions and rotations of all objects, all object positions are relative to the gripper position
  * Goals: Desired positions of objects
  * Reward Function

### Next steps
* TODO
