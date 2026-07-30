# environmental

## Purpose


## Methods/Exposes

### `enviorment.py`

- `spawn_drone` - creates a pybullet body matching the drone mass/inertia
 and returns the body_id of the pybullet object
  
- `sample_wind_conditions` - calculated wind resistance on each of the axis(roll,pitch,yaw) and mass_scale of the object
- `apply_rotor_thrust` - method to apply the calculated combined thrust on the pybullet object
- `apply_rotor_torque` - method to apply the torque on the pybullet object

### `interceptor_drone.py`

- `build_observation` - mapping the drone state values into the input shape
- `InterceptorDroneEnv`
  - `_get_obs` - starting the observable values
  - `_compute_reward` - calculating the reward/penalty for the drone based on it current state
  - `__init__` - build the setup for the training env , setups the dt,max_steps,the render action and observable 
  - `reset` - resets the current state so the rl could learn with new params
  - `step` - doing a training step each time 
  - `render` - displays the training in pybullet
- `my_check_env` - checks that the overall class setup is valid to the gymnasium requirements
- `my_test` - tests that the rl config actually can learn

## Depends on

`pybullet` , `gymnasium` , `numpy` , `scipy` , `drone.py` , `methods.py` , `enviorment.py` (used by `interceptor_drone.py`)


## Notes
#### unused
`apply_rotor_torque` and `apply_rotor_thrust` are still unused will be used when the rl environment is ready
### todo
implement `render`