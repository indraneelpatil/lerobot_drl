#!/usr/bin/env python
"""
Real robot data collection script for SO101 follower + leader.

Usage:
    python -m neel.robot.real_robot_manipulator --config_path neel/simulation/config/real_robot_env_record.json
"""

from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.teleoperators import (
    gamepad,  # noqa: F401
    keyboard,  # noqa: F401
    make_teleoperator_from_config,
)

import logging
from dataclasses import dataclass
from typing import Any
import time
import torch
import numpy as np

import gymnasium as gym
from lerobot.configs import parser
from lerobot.envs.configs import HILSerlRobotEnvConfig
from lerobot.processor import (
    DataProcessorPipeline,
    EnvTransition,
    create_transition,
    TransitionKey,
)
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.constants import ACTION, DONE, OBS_IMAGES, OBS_STATE, REWARD
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Import from isaac_gym_utils (which now has real_robot support)
import sys
sys.path.append('/Users/sisaha/Documents/Personal dev/physicalaihack/lerobot_drl')
from neel.simulation.isaac_gym_utils import (
    make_processors,
    make_robot_env,
    step_env_and_process_transition,
)

logging.basicConfig(level=logging.INFO)

hc_joint_names = ["shoulder_pan",
                  "shoulder_lift",
                  "elbow_flex",
                  "wrist_flex",
                  "wrist_roll",
                  "gripper"]

@dataclass
class DatasetConfig:
    """Configuration for dataset creation and management."""

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


def control_loop(
    env: gym.Env,
    env_processor: DataProcessorPipeline[EnvTransition, EnvTransition],
    action_processor: DataProcessorPipeline[EnvTransition, EnvTransition],
    cfg: GymManipulatorConfig
):
    """Main control loop for real robot interaction."""

    dt = 1.0 / cfg.env.fps

    print(f"Starting real robot control loop at {cfg.env.fps} FPS")
    print("Controls:")
    print("- Use leader arm for intervention")
    print("- When not intervening, robot will stay still")
    print("- Press 's' for success, 'f' for failure, 'r' to rerecord")
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

    print("#" * 50)
    print("Starting real robot control loop")
    print("#" * 50)

    # Main control loop
    while episode_idx < cfg.dataset.num_episodes_to_record:
        step_start_time = time.perf_counter()

        # Create a neutral action (robot stays still unless intervened)
        neutral_action = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
        if use_gripper:
            neutral_action = torch.cat([neutral_action, torch.tensor([1.0])])  # Gripper stay

        # Execute step with processor pipeline
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
                k: v.squeeze(0).cpu() if isinstance(v, torch.Tensor) else v
                for k, v in transition[TransitionKey.OBSERVATION].items()
            }

            action = transition[TransitionKey.ACTION].squeeze(0).cpu() if isinstance(transition[TransitionKey.ACTION], torch.Tensor) else transition[TransitionKey.ACTION]
            reward = transition[TransitionKey.REWARD].squeeze(0).cpu() if isinstance(transition[TransitionKey.REWARD], torch.Tensor) else transition[TransitionKey.REWARD]
            done = torch.tensor([terminated or truncated], dtype=torch.bool)

            # Add to dataset
            dataset_frame = {
                **observations,
                ACTION: action,
                REWARD: reward,
                DONE: done,
            }

            # Add discrete penalty if using gripper
            if use_gripper and "discrete_penalty" in transition[TransitionKey.COMPLEMENTARY_DATA]:
                discrete_penalty = transition[TransitionKey.COMPLEMENTARY_DATA]["discrete_penalty"]
                discrete_penalty = discrete_penalty.squeeze(0).cpu() if isinstance(discrete_penalty, torch.Tensor) else discrete_penalty
                dataset_frame["complementary_info.discrete_penalty"] = discrete_penalty

            dataset.add_frame(dataset_frame)

        episode_step += 1

        # Handle episode termination
        if terminated or truncated:
            episode_duration = time.perf_counter() - episode_start_time
            logging.info(f"Episode {episode_idx + 1} finished in {episode_step} steps ({episode_duration:.2f}s)")

            if cfg.mode == "record":
                dataset.save_episode()
                logging.info(f"Saved episode {episode_idx + 1} to dataset")

            episode_idx += 1
            episode_step = 0

            if episode_idx < cfg.dataset.num_episodes_to_record:
                # Reset for next episode
                logging.info(f"Starting episode {episode_idx + 1}/{cfg.dataset.num_episodes_to_record}")
                obs, info = env.reset()
                complimentary_data = (
                    {"raw_joint_positions": info.pop("raw_joint_positions")} if "raw_joint_positions" in info else {}
                )
                env_processor.reset()
                action_processor.reset()

                transition = create_transition(observation=obs, info=info, complementary_data=complimentary_data)
                transition = env_processor(data=transition)
                episode_start_time = time.perf_counter()

        # Maintain FPS
        step_duration = time.perf_counter() - step_start_time
        busy_wait(dt - step_duration)

    # Push to HuggingFace Hub if configured
    if cfg.mode == "record" and cfg.dataset.push_to_hub:
        logging.info("Pushing dataset to HuggingFace Hub...")
        dataset.push_to_hub()
        logging.info("Dataset pushed successfully")

    logging.info("Control loop completed")


@parser.wrap()
def main(cfg: GymManipulatorConfig):
    """Main entry point for real robot data collection."""

    logging.info("=" * 50)
    logging.info("Real Robot Data Collection")
    logging.info("=" * 50)

    # Validate configuration
    if cfg.env.name != "real_robot":
        raise ValueError(f"This script is for real_robot environment, got: {cfg.env.name}")

    # Create environment and teleop
    logging.info("Creating real robot environment...")
    env, teleop_device = make_robot_env(cfg.env, device=cfg.device)

    # Create processors
    logging.info("Creating processor pipelines...")
    env_processor, action_processor = make_processors(
        env, teleop_device, cfg.env, cfg.device
    )

    try:
        # Run control loop
        control_loop(env, env_processor, action_processor, cfg)
    except KeyboardInterrupt:
        logging.info("Interrupted by user")
    finally:
        # Cleanup
        logging.info("Cleaning up...")
        env.close()
        if teleop_device is not None and hasattr(teleop_device, "disconnect"):
            teleop_device.disconnect()

    logging.info("Finished!")


if __name__ == "__main__":
    main()
