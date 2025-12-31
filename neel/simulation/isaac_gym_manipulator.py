# !/usr/bin/env python
"""
Created by Indraneel on 12/29/2025
"""


"""Launch Isaac Sim Simulator first."""
import multiprocessing
if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)

from isaaclab.app import AppLauncher

# launch omniverse app
app_launcher = AppLauncher()
simulation_app = app_launcher.app


"""Rest everything else follows."""

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.managers import TerminationTermCfg, DatasetExportMode

import logging
from dataclasses import dataclass
from typing import Any
import time
import torch
import numpy as np

import gymnasium as gym
from lerobot.configs import parser
from lerobot.envs.configs import HILSerlRobotEnvConfig
from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DataProcessorPipeline,
    DeviceProcessorStep,
    EnvTransition,
    InterventionActionProcessorStep,
    Torch2NumpyActionProcessorStep,
    Numpy2TorchActionProcessorStep,
    VanillaObservationProcessorStep,
    create_transition,
    TransitionKey
)
from lerobot.processor.converters import identity_transition
from lerobot.utils.robot_utils import busy_wait
from lerobot.teleoperators import (
    keyboard
)
from lerobot.utils.constants import ACTION, DONE, OBS_IMAGES, OBS_STATE, REWARD
from lerobot.datasets.lerobot_dataset import LeRobotDataset

logging.basicConfig(level=logging.INFO)

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

def make_robot_env(cfg: HILSerlRobotEnvConfig) -> tuple[gym.Env, Any]:
    """Create robot environment
    
    """
    if cfg.name == "gym_hil":
        assert cfg.robot is None and cfg.teleop is None, "GymHIL environment does not support robot or teleop"
        import gym_hil

        # Extract gripper settings
        use_gripper = cfg.processor.gripper.use_gripper if cfg.processor.gripper is not None else True
        gripper_penalty = cfg.processor.gripper.gripper_penalty if cfg.processor.gripper is not None else 0.0

        env = gym.make(
            f"gym_hil/{cfg.task}",
            image_obs=True,
            render_mode="human",
            use_gripper=use_gripper,
            gripper_penalty=gripper_penalty,
            random_block_position=True,
        )

        return env, None
    
    if cfg.name == "gym_isaac_sim_hil":
        pass


    # TODO Real robot environment
    raise NotImplementedError("Real robot environment not implemented yet")


@parser.wrap()
def main(cfg: GymManipulatorConfig) -> None:
    """ Main entry """
    # env, teleop_device = make_robot_env(cfg.env)

    # Close the simulator
    simulation_app.close()


if __name__ == "__main__":
    main()