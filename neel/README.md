# DRL Simulation

The implementation is an algorithm from this paper: https://hil-serl.github.io/. There are two simulations implemented in this folder, one with the FrankaPy robot in Mujoco (this is the default one which is in the lerobot repository) and a new one with lerobot in Isaac Sim.

## Mujoco Setup

* Follow the lerobot instructions to setup lerobot repo in your conda environment
* Follow the instructions here https://huggingface.co/docs/lerobot/en/hilserl_sim to setup the Mujoco environment

### Collect Data

```bash
python simulation/gym_manipulator.py --config_path simulation/config/gym_hil_env_record.json
```

### Run Training

Run Actor
```bash
python -m simulation.actor --config_path simulation/config/gym_hil_env_train.json
```


Run Learner
```bash
python -m simulation.learner --config_path simulation/config/gym_hil_env_train.json
```

## Isaac Sim Setup

* Install Isaac Sim in the same conda environment as the lerobot packages
* Install the Isaac Sim gym environment from https://github.com/LightwheelAI/leisaac by pip installing this package in the conda environment. More instructions on this are here : https://wiki.seeedstudio.com/simulate_soarm101_by_leisaac/
* The learner is the same script but the actor and data collection script now use the isaac sim gym environment


### Isaac Sim Environment Intro
* Action Space: Joint angles in radians for all the 6 joints ["shoulder_pan", "shoulder_lift" "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
* Observation Space: Joint angles for these joints, camera images for wrist and front camera


### Collect Data

```bash
python -m simulation.isaac_gym_manipulator --config_path simulation/config/leisaac_env_record.json
```

### Run Training

Run Actor
```bash
python -m simulation.isaac_gym_actor --config_path simulation/config/isaac_gym_env_train.json
```


Run Learner
```bash
python -m simulation.learner --config_path simulation/config/gym_hil_env_train.json
```

