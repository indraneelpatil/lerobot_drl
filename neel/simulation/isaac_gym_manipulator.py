# !/usr/bin/env python
"""
Created by Indraneel on 12/29/2025
"""


"""Launch Isaac Sim Simulator first."""
import multiprocessing
if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)

from isaaclab.app import AppLauncher

# isaac sim args
app_launcher_args = {
    "enable_cameras" : True
}

# launch omniverse app
app_launcher = AppLauncher(app_launcher_args)
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
    GripperPenaltyProcessorStep
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
        if hasattr(env_cfg.terminations, "time_out"):
            env_cfg.terminations.time_out = None
        if hasattr(env_cfg.terminations, "success"):
            env_cfg.terminations.success = None
        
        # TODO recording
        env_cfg.recorders = None
        
        # Create environment
        env: ManagerBasedRLEnv = gym.make(cfg.task, cfg=env_cfg)

        assert cfg.teleop is not None, "Teleop config must be provided for gym isaac environment"
        teleop_device = make_teleoperator_from_config(cfg.teleop)
        teleop_device.connect()

        return env, teleop_device

    # TODO Real robot environment
    raise NotImplementedError("Real robot environment not implemented yet")

# NOTE: Neel can you check joint pos is in the order of shoulder -> gripper
def get_raw_joint_positions(
    obs: dict[str, Any],
    joint_names: list[str],
) -> dict[str, float]:
    policy_obs = obs["policy"]
    joint_pos = policy_obs["joint_pos"]  # shape: (B, N)

    # assume batch size = 1
    joint_pos = joint_pos[0]

    assert len(joint_names) == joint_pos.shape[0], (
        f"Expected {len(joint_names)} joints, got {joint_pos.shape[0]}"
    )

    return {
        f"{name}.pos": joint_pos[i].item()
        for i, name in enumerate(joint_names)
    }



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

    raw_joint_positions = get_raw_joint_positions(transition[TransitionKey.OBSERVATION], hc_joint_names)
    # Create action transition
    transition[TransitionKey.ACTION] = action
    transition[TransitionKey.OBSERVATION] = raw_joint_positions

    processed_action_transition = action_processor(transition)
    processed_action = processed_action_transition[TransitionKey.ACTION]

    obs, reward, terminated, truncated, info = env.step(processed_action)


    reward = reward + processed_action_transition[TransitionKey.REWARD]
    terminated = terminated or processed_action_transition[TransitionKey.DONE]
    truncated = truncated or processed_action_transition[TransitionKey.TRUNCATED]
    complementary_data= processed_action_transition[TransitionKey.COMPLEMENTARY_DATA]
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
    # print(f"Observation is {obs}")
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

        if cfg.mode == "record":
            observations = {
                k: v.squeeze(0).cpu() 
                for k,v in transition[TransitionKey.OBSERVATION].items()
                if isinstance(v, torch.Tensor)
            }
            # Use teleop_action if available, otherwise use the action from the transition
            action_to_record = transition[TransitionKey.INFO].get(
                "teleop_action", transition[TransitionKey.ACTION]
            )
            frame = {
                **observations,
                ACTION: action_to_record.squeeze(0).cpu() if isinstance(action_to_record, torch.Tensor) else action_to_record.astype(np.float32),
                REWARD: transition[TransitionKey.REWARD].cpu() if isinstance(transition[TransitionKey.REWARD], torch.Tensor) else np.array([transition[TransitionKey.REWARD]], dtype=np.float32),
                DONE: np.array([terminated or truncated], dtype=bool)
            }
            if use_gripper:
                discrete_penalty = transition[TransitionKey.INFO].get("discrete_penalty", 0.0)
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

    # TODO Push to hub


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

        env_pipeline_steps = [VanillaObservationProcessorStep()]

        # TODO Add Joint velocities
        # TODO Add motor current

        if kinematics_solver is not None:
            env_pipeline_steps.append(
                ForwardKinematicsJointsToEEObservation(
                    kinematics=kinematics_solver,
                    motor_names=joint_names,
                )
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
    simulation_app.close()


if __name__ == "__main__":
    main()