# !/usr/bin/env python
"""
Created by Indraneel on 01/27/2025
"""

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
    AddBatchDimensionProcessorStep,
    AddTeleopActionAsComplimentaryDataStep,
    AddTeleopEventsAsInfoStep,
    DataProcessorPipeline,
    DeviceProcessorStep,
    EnvTransition,
    InterventionActionProcessorStep,
    Torch2NumpyActionProcessorStep,
    Numpy2TorchActionProcessorStep,
    VanillaObservationProcessorStep,
    create_transition,
    TransitionKey,
    MapTensorToDeltaActionDictStep,
    MapDeltaActionToRobotActionStep,
    RobotActionToPolicyActionProcessorStep,
    GripperPenaltyProcessorStep,
    Degrees2RadiansActionProcessorStep,
    Radians2DegreesObservationProcessor,
    AddProcessorObservationsToState
)
from lerobot.robots.so100_follower.robot_kinematic_processor import (
    EEBoundsAndSafety,
    EEReferenceAndDelta,
    ForwardKinematicsJointsToEEObservation,
    GripperVelocityToJoint,
    InverseKinematicsRLStep,
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

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.utils import parse_env_cfg

hc_joint_names = ["shoulder_pan",
                  "shoulder_lift",
                  "elbow_flex",
                  "wrist_flex",
                  "wrist_roll",
                  "gripper"]


def make_robot_env(cfg: HILSerlRobotEnvConfig, device: str) -> tuple[gym.Env, Any]:
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
    
    if cfg.name == "gym_isaac_sim_hil" and cfg.task !=None :
        env_cfg = parse_env_cfg(cfg.task, device, num_envs=1)
        env_cfg.use_teleop_device(cfg.processor.control_mode)
        env_cfg.seed = int(time.time())

        # Modify configuration
        # if hasattr(env_cfg.terminations, "time_out"):
        #     env_cfg.terminations.time_out = None
        # if hasattr(env_cfg.terminations, "success"):
        #     env_cfg.terminations.success = None
        
        # TODO recording
        env_cfg.recorders = None
        
        # Create environment
        env: ManagerBasedRLEnv = gym.make(cfg.task, cfg=env_cfg)

        assert cfg.teleop is not None, "Teleop config must be provided for gym isaac environment"
        teleop_device = make_teleoperator_from_config(cfg.teleop)
        teleop_device.connect()

        # Wrap the environment
        env_wrapped = IsaacSimEnvWrapper(env)

        return env_wrapped, teleop_device

    # TODO Real robot environment
    raise NotImplementedError("Real robot environment not implemented yet")



class IsaacSimEnvWrapper(gym.Wrapper):
    def __init__(
        self,
        env,):
        """Init the wrapper"""
        super().__init__(env)
        self._raw_joint_positions = None

    def convert_joint_angle_tensor_to_dict(self, joint_pos):
        joint_pos = joint_pos[0]

        assert len(hc_joint_names) == joint_pos.shape[0], (
            f"Expected {len(hc_joint_names)} joints, got {joint_pos.shape[0]}"
        )

        return {
            f"{name}.pos": joint_pos[i].item()
            for i, name in enumerate(hc_joint_names)
        }
    
    def _get_observation(self, raw_obs):
        # Update raw joint positions
        policy_obs = raw_obs["policy"]
        joint_pos = policy_obs["joint_pos"]  # shape: (B, N)

        joint_angle_dict = self.convert_joint_angle_tensor_to_dict(joint_pos)
        self._raw_joint_positions = joint_angle_dict
        joint_positions = np.array([joint_angle_dict[f"{name}.pos"] for name in hc_joint_names])

        # Process images
        camera_keys = ["front", "wrist"]
        images = {key: policy_obs[key] for key in camera_keys}

        return {"agent_pos": joint_positions, "pixels": images, **joint_angle_dict}

    def step(self, action):
        # Step the environment
        obs, reward, terminated, truncated, info = self.env.step(action)

        lerobot_obs = self._get_observation(obs)

        return lerobot_obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        """Reset the environment."""
        # Update raw joint positions

        obs, info = self.env.reset(**kwargs)
        lerobot_obs = self._get_observation(obs)

        return lerobot_obs, info
    
    def get_raw_joint_positions(self) -> dict[str, float]:
        """Get raw joint positions in degrees."""

        raw_joint_positions_deg = {}
        for k,v in self._raw_joint_positions.items():
            raw_joint_positions_deg[k] = np.rad2deg(v)
        return raw_joint_positions_deg


def step_env_and_process_transition(
    env: gym.Env,
    transition: EnvTransition,
    action: torch.Tensor,
    env_processor: DataProcessorPipeline[EnvTransition, EnvTransition],
    action_processor: DataProcessorPipeline[EnvTransition, EnvTransition],
) -> EnvTransition:
    """
    Execute one step with processor pipeline
    """

    # Create action transition
    transition[TransitionKey.ACTION] = action
    transition[TransitionKey.OBSERVATION] = (
        env.get_raw_joint_positions() if hasattr(env, "get_raw_joint_positions") else {}
    )

    processed_action_transition = action_processor(transition)
    processed_action = processed_action_transition[TransitionKey.ACTION]

    obs, reward, terminated, truncated, info = env.step(processed_action)


    reward = reward + processed_action_transition[TransitionKey.REWARD]
    terminated = terminated or processed_action_transition[TransitionKey.DONE]
    truncated = truncated or processed_action_transition[TransitionKey.TRUNCATED]
    complementary_data= processed_action_transition[TransitionKey.COMPLEMENTARY_DATA].copy()
    new_info = processed_action_transition[TransitionKey.INFO].copy()
    new_info.update(info)

    new_transition = create_transition(
        observation=obs,
        action=action,
        reward=reward,
        done=terminated,
        truncated=truncated,
        info=new_info,
        complementary_data=complementary_data
    )
    new_transition = env_processor(new_transition)

    return new_transition

def make_processors(
        env: gym.Env, teleop_device: Teleoperator | None, cfg: HILSerlRobotEnvConfig, device: str ="cpu"
) -> tuple[DataProcessorPipeline[EnvTransition, EnvTransition], DataProcessorPipeline[EnvTransition, EnvTransition]]:
    """Create action processors"""
    terminate_on_success = (
        cfg.processor.reset.terminate_on_success if cfg.processor.reset is not None else True
    )

    joint_names = hc_joint_names
    print(f"Joint names are {joint_names}")

    # Set up kinematics solver if inverse kinematics is configured
    kinematics_solver = None
    if cfg.processor.inverse_kinematics is not None:
        kinematics_solver = RobotKinematics(
            urdf_path=cfg.processor.inverse_kinematics.urdf_path,
            target_frame_name=cfg.processor.inverse_kinematics.target_frame_name,
            joint_names=joint_names,
        )

    if cfg.name == "gym_isaac_sim_hil":
        action_pipeline_steps = [
            AddTeleopActionAsComplimentaryDataStep(teleop_device=teleop_device),
            AddTeleopEventsAsInfoStep(teleop_device=teleop_device),
            InterventionActionProcessorStep(
                use_gripper=cfg.processor.gripper.use_gripper if cfg.processor.gripper is not None else False,
                terminate_on_success=terminate_on_success),
        ]

        # Replace InverseKinematicsProcessor with new kinematic processors
        if cfg.processor.inverse_kinematics is not None and kinematics_solver is not None:
            # Add EE bounds and safety processor
            inverse_kinematics_steps = [
                MapTensorToDeltaActionDictStep(
                    use_gripper=cfg.processor.gripper.use_gripper if cfg.processor.gripper is not None else False
                ),
                MapDeltaActionToRobotActionStep(),
                EEReferenceAndDelta(
                    kinematics=kinematics_solver,
                    end_effector_step_sizes=cfg.processor.inverse_kinematics.end_effector_step_sizes,
                    motor_names=joint_names,
                    use_latched_reference=False,
                    use_ik_solution=True,
                ),
                EEBoundsAndSafety(
                    end_effector_bounds=cfg.processor.inverse_kinematics.end_effector_bounds,
                ),
                GripperVelocityToJoint(
                    clip_max=cfg.processor.max_gripper_pos,
                    speed_factor=1.0,
                    discrete_gripper=True,
                ),
                InverseKinematicsRLStep(
                    kinematics=kinematics_solver, motor_names=joint_names, initial_guess_current_joints=False
                ),
            ]
            action_pipeline_steps.extend(inverse_kinematics_steps)
            action_pipeline_steps.append(RobotActionToPolicyActionProcessorStep(motor_names=joint_names))
            action_pipeline_steps.append(AddBatchDimensionProcessorStep())
            action_pipeline_steps.append(Degrees2RadiansActionProcessorStep())

        env_pipeline_steps = [
            Radians2DegreesObservationProcessor(),
            VanillaObservationProcessorStep()]


        # TODO Add Joint velocities
        # TODO Add motor current

        if kinematics_solver is not None:
            env_pipeline_steps.append(
                ForwardKinematicsJointsToEEObservation(
                    kinematics=kinematics_solver,
                    motor_names=joint_names,
                )
        )
        env_pipeline_steps.append(
            AddProcessorObservationsToState()
        )
            
        # TODO Consider adding image preprocessing
        # TODO Consider adding time limit processor

        # Add gripper penalty processor if gripper config exists and enabled
        if cfg.processor.gripper is not None and cfg.processor.gripper.use_gripper:
            env_pipeline_steps.append(
                GripperPenaltyProcessorStep(
                    penalty=cfg.processor.gripper.gripper_penalty,
                    max_gripper_pos=cfg.processor.max_gripper_pos,
                )
            )

        #note: Skipped reward classifier
        

        env_pipeline_steps.append(AddBatchDimensionProcessorStep())
        env_pipeline_steps.append(DeviceProcessorStep(device=device))
    
        return DataProcessorPipeline(
            steps=env_pipeline_steps, to_transition=identity_transition, to_output=identity_transition
        ), DataProcessorPipeline(steps=action_pipeline_steps, to_transition=identity_transition, to_output=identity_transition)

    raise NotImplementedError("Real robot processors not implemented")