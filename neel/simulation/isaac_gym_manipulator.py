# !/usr/bin/env python
"""
Created by Indraneel on 12/29/2025

python -m simulation.isaac_gym_manipulator --config_path simulation/config/leisaac_env_record.json
"""


"""Launch Isaac Sim Simulator first."""
from isaaclab.app import AppLauncher

# isaac sim args
app_launcher_args = {
    "enable_cameras" : True
}

# launch omniverse app
app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app


"""Rest everything else follows."""


import logging
from dataclasses import dataclass
from typing import Any
import time
import torch
import numpy as np
from pprint import pprint

import leisaac
import gymnasium as gym
from lerobot.configs import parser
from lerobot.envs.configs import HILSerlRobotEnvConfig
from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import (
    DataProcessorPipeline,
    DeviceProcessorStep,
    EnvTransition,
    create_transition,
    TransitionKey,
)
from lerobot.processor.converters import identity_transition
from lerobot.utils.robot_utils import busy_wait
from lerobot.teleoperators import (
    gamepad,  # noqa: F401
    keyboard,  # noqa: F401
    make_teleoperator_from_config,
    so101_leader,  # noqa: F401
)
from lerobot.utils.constants import ACTION, DONE, OBS_IMAGES, OBS_STATE, REWARD
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from .isaac_gym_utils import (
    create_transition,
    make_processors,
    make_robot_env,
    step_env_and_process_transition,
)

logging.basicConfig(level=logging.INFO)

# Teleop action features are {'shoulder_pan.pos': <class 'float'>, 'shoulder_lift.pos': <class 'float'>, 'elbow_flex.pos': <class 'float'>, 'wrist_flex.pos': <class 'float'>, 'wrist_roll.pos': <class 'float'>, 'gripper.pos': <class 'float'>}
hc_joint_names = ["shoulder_pan",
                  "shoulder_lift",
                  "elbow_flex",
                  "wrist_flex",
                  "wrist_roll",
                  "gripper"]

@dataclass
class DatasetConfig:
    """ Configuration for dataset creation and management."""

    repo_id: str
    task: str
    root: str | None = None
    num_episodes_to_record: int = 5
    replay_episode: int | None = None
    push_to_hub: bool = False


@dataclass
class GymManipulatorConfig:

    env: HILSerlRobotEnvConfig
    dataset: DatasetConfig
    mode: str | None = None
    device: str = "cpu"



def control_loop(env: gym.Env,
                 env_processor: DataProcessorPipeline[EnvTransition, EnvTransition],
                 action_processor: DataProcessorPipeline[EnvTransition, EnvTransition],
                cfg: GymManipulatorConfig):
    """ Main control loop for environment interaction"""

    dt = 1.0 / cfg.env.fps

    print(f"Starting control loop at {cfg.env.fps} FPS")
    print("Controls:")
    print("- Use teleop device for intervenion")
    print("- When not intervening, robot will stay still")
    print("- Press Ctrl+C to exit")

    # Reset environment
    obs, info = env.reset()
    complimentary_data = (
        {"raw_joint_positions": info.pop("raw_joint_positions")} if "raw_joint_positions" in info else {}
    )
    env_processor.reset()
    action_processor.reset()

    # Process initial observation
    transition = create_transition(observation=obs, info=info, complementary_data=complimentary_data)
    transition = env_processor(data=transition)

    # Check gripper
    use_gripper = cfg.env.processor.gripper.use_gripper if cfg.env.processor.gripper is not None else True

    dataset = None
    if cfg.mode == "record":
        #action_features = teleop_device.action_features
        features = {
            ACTION: {"dtype": "float32", "shape": (4,), "names": None},
            REWARD: {"dtype": "float32", "shape": (1,), "names": None},
            DONE: {"dtype": "bool", "shape": (1,), "names": None},
        }
        if use_gripper:
            features["complementary_info.discrete_penalty"] = {
                "dtype": "float32",
                "shape": (1,),
                "names": ["discrete_penalty"],
            }

        for key, value in transition[TransitionKey.OBSERVATION].items():
            if key == OBS_STATE:
                features[key] = {
                    "dtype": "float32",
                    "shape": value.squeeze(0).shape,
                    "names": None
                }
            if "image" in key:
                features[key] = {
                    "dtype": "video",
                    "shape": value.squeeze(0).shape,
                    "names": ["channels", "height", "width"] 
                }

        # Create dataset
        dataset = LeRobotDataset.create(
            cfg.dataset.repo_id,
            cfg.env.fps,
            root=cfg.dataset.root,
            use_videos=True,
            image_writer_threads=4,
            image_writer_processes=0,
            features=features
        )

    episode_idx = 0
    episode_step = 0
    episode_start_time = time.perf_counter()

    print("############################## Starting control loop")
    # simulate environment
    while episode_idx < cfg.dataset.num_episodes_to_record and simulation_app.is_running():
        step_start_time = time.perf_counter()
        
        # Create a neutral action 
        neutral_action = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
        if use_gripper:
            neutral_action = torch.cat([neutral_action, torch.tensor([1.0])]) # Gripper stay

        # Use the new step function
        transition = step_env_and_process_transition(
            env=env,
            transition=transition,
            action=neutral_action,
            env_processor=env_processor,
            action_processor=action_processor,
        )
        terminated = transition.get(TransitionKey.DONE, False)
        truncated = transition.get(TransitionKey.TRUNCATED, False)

        terminated = terminated.squeeze(0).cpu() if isinstance(terminated, torch.Tensor) else terminated
        truncated = truncated.squeeze(0).cpu() if isinstance(truncated, torch.Tensor) else truncated

        if cfg.mode == "record":
            observations = {
                k: v.squeeze(0).cpu() 
                for k,v in transition[TransitionKey.OBSERVATION].items()
                if isinstance(v, torch.Tensor)
            }
            # Use teleop_action if available, otherwise use the action from the transition
            action_to_record = transition[TransitionKey.COMPLEMENTARY_DATA].get(
                "teleop_action", transition[TransitionKey.ACTION]
            )
            frame = {
                **observations,
                ACTION: action_to_record.squeeze(0).cpu() if isinstance(action_to_record, torch.Tensor) else action_to_record.astype(np.float32),
                REWARD: transition[TransitionKey.REWARD].cpu() if isinstance(transition[TransitionKey.REWARD], torch.Tensor) else np.array([transition[TransitionKey.REWARD]], dtype=np.float32),
                DONE: np.array([terminated or truncated], dtype=bool)
            }
            if use_gripper:
                discrete_penalty = transition[TransitionKey.COMPLEMENTARY_DATA].get("discrete_penalty", 0.0)
                frame["complementary_info.discrete_penalty"] = np.array([discrete_penalty], dtype=np.float32)
            
            if dataset is not None:
                frame["task"] = cfg.dataset.task
                dataset.add_frame(frame)

        episode_step += 1

        # Handle termination
        if terminated or truncated:
            episode_time = time.perf_counter() - episode_start_time
            print(
                f"Episode ended after {episode_step} steps in {episode_time:.1f}s with reward {transition[TransitionKey.REWARD]}"            
            )
            episode_step = 0
            episode_idx += 1

            if dataset is not None:
                if transition[TransitionKey.INFO].get("rerecord_episode", False):
                    print(f"Re-recording episode {episode_idx}")
                    dataset.clear_episode_buffer()
                    episode_idx -= 1
                else:
                    print(f"Saving episode {episode_idx}")
                    dataset.save_episode()

            # Reset for new episode
            obs, info = env.reset()
            env_processor.reset()
            action_processor.reset()


            transition = create_transition(observation=obs, info=info)
            transition = env_processor(transition)

        # Maintain fps timing
        busy_wait(dt - (time.perf_counter() - step_start_time))

    if dataset is not None and cfg.dataset.push_to_hub:
        logging.info("Pushing dataset to hub")
        dataset.push_to_hub()



@parser.wrap()
def main(cfg: GymManipulatorConfig) -> None:
    """ Main entry """
    env, teleop_device = make_robot_env(cfg.env, cfg.device)
    env_processor, action_processor = make_processors(env, teleop_device, cfg.env, cfg.device)

    print("Environment observation space:", env.observation_space)
    print("Environment Action Space:", env.action_space)
    print("Environment processor:", env_processor)
    print("Action processor:", action_processor)

    control_loop(env, env_processor, action_processor, cfg)

    # Close the simulator
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()