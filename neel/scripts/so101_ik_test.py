# !/usr/bin/env python
"""
Created by Indraneel on 01/10/2026

Inverse Kinematics Test for So101 arm
"""

import argparse
from typing import Any
import numpy as np

from lerobot.model.kinematics import RobotKinematics



def main(args_cli: Any):

    hc_joint_names = ["shoulder_pan",
                  "shoulder_lift",
                  "elbow_flex",
                  "wrist_flex",
                  "wrist_roll",
                  "gripper"]

    # Create kinematics solver
    kinematics_solver = RobotKinematics(
        urdf_path=args_cli.urdf_path,
        target_frame_name=args_cli.target_frame_name,
        joint_names=hc_joint_names,
    )

    # Initial position joint angles in degrees
    j0 = np.array([0.0, -30.0, 60.0, -30.0, 0.0, 0.0])

    # Forward kinematics
    t_curr = kinematics_solver.forward_kinematics(j0)
    print(f"t_curr is {t_curr}")

    # Inverse kinematics in degrees
    j0_inv = kinematics_solver.inverse_kinematics(j0, t_curr)
    print(f"j0_inv {j0_inv}")
    assert np.allclose(j0_inv, j0, rtol=1e-6, atol=1e-8)

    # Perturb t_curr a little bit
    delta = np.array([0.0, 0.0, 0.1])
    t_curr[:3,3] += delta 
    print(f"t_curr {t_curr}")

    # Try again
    j1_inv = kinematics_solver.inverse_kinematics(j0_inv, t_curr)
    print(f"j1_inv {j1_inv}")
    # assert np.allclose(j0_inv, j0, rtol=1e-6, atol=1e-8)

    # forward kinematics
    t_new = kinematics_solver.forward_kinematics(j1_inv)
    print(f"t_new is {t_new}")
    


if __name__ == "__main__":
    # add argparse arguments
    parser = argparse.ArgumentParser(description="leisaac teleoperation for leisaac environments.")
    parser.add_argument("--urdf_path", type=str, default='/home/neel/Projects/lerobot_drl/assets/robots/so101.urdf', help="Urdf path for so101 follower")
    parser.add_argument("--target_frame_name", type=str, default='gripper_frame_link', help="Kinematics target frame")

    # Parse the arguments
    args_cli = parser.parse_args()
    main(args_cli)