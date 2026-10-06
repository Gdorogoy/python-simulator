# demo

## Purpose
Showing an demo of the project and how the drone moves

## Methods/Exposes

### `demo.py`

- `angle_diff` - calculates the difference between two angles in (-pi -> pi) range , (signed)
- `altitude_thrust` -  calculated the thrust magnitude which is calculated from thrust altitude error ; 
- `attitude_torque` - computes the error as an quaternion rotation between current and target
- `position_to_attitude` - calculates the needed target_roll and target_pitch based on the attitude error (target.x/y vs state.x/y)
- `simulation` - displaying the capabilities of the drone with all of the physics enabled

## Depends on
`numpy`,`pybullet`,`scipy`,`drone.py`,`methods.py`,`enviorment.py`,`pybullet_data`

## Notes
`altitude_torque` - computed in quaternions as the the euler calculation can get corrupted by roll/pitch/yaw because they can couple when yaw is spinning to fast  
