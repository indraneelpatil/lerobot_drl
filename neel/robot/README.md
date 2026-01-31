# Porting HIL-SERL to the robot

1. New scripts to replace isaac sim environment with real robot 
2. Add leader arm teleop (currently it is keyboard)
2. Reward Classifer to know if the task succeeded or not (We can also do this manually using keyboard)
3. Add End Affector workspace bounds to make sure robot doesnt collide with anything
4. Readd the max_ee_step_m check to 0.05m in the processor
5. Set up Cameras, crop images in the environment processors
6. Consider adding extra observations like velocities
7. Tune hyperparameters
8. Consider reusing data from simulation
9. Write an inference script which loads and runs the trained model