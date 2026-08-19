# app

## Features
- Simulating real-world physics forces applied on the drone
- Training an interceptor drone to reach a moving target using reinforcement learning (PPO)
- Training an evader drone to evade the interceptor and hit the land target (PPO)
- Estimating drone state from noisy/delayed sensor readings, separate from the true physics state
- Validating the physics/config setup with automated checks
- A choreographed flight demo showing the physics engine driving the drone through a flight plan in PyBullet

## Blocks
- **dynamics** - the physics engine: drone structure (rotors, frame, state) and the force/torque math that steps the drone forward in time
- **environmental** - the glue between the physics engine and the outside world: spawns the drone in PyBullet, applies wind, and wraps everything as a Gymnasium RL environment for the interceptor
- **control** - the classical PID hover controller: used standalone as a baseline, tuned via Optuna, and as the teacher whose demonstrations pretrain the RL policy via behavior cloning before PPO takes over
- **reward_functions** - builds the reward function passed to the env: shared termination checks plus a chainable curriculum of reward shapes (imitation -> phase 0 -> base)
- **guidance** - the learning block: an actor-critic network pretrained via behavior cloning on the PID's demonstrations, then trained further with PPO so the interceptor learns to reach its target
- **navigation** - reads the true state from dynamics (read-only, never writes to it), adds sensor noise/delay/drift, and outputs an estimated state; a one-way transform, not part of the physics update (not currently wired into the env's observations - self-contained estimator with its own test)
- **demo** - runs a scripted flight plan (altitude changes, tilts, translation, circling, yaw spin) through the physics engine to showcase what the drone can do
- **test** - checks that a given drone configuration is physically valid (mixer invertibility, hover equilibrium, stability, etc.)
- **main.py** - entry point; menu to run the demo, the tests, or the RL training/interceptor

## Folder structure
```
app/
├── main.py             entry point (demo / test / RL menu)
├── dynamics/            physics engine (drone structure + force calculations)
├── environmental/       PyBullet glue + Gymnasium interceptor environment
├── control/             PID hover controller + tuning + BC demonstration pipeline
├── reward_functions/    reward-function construction (termination + shaping curriculum)
├── guidance/            behavior cloning + PPO training (actor-critic)
├── navigation/          read-only state estimator (sensor noise/delay/drift)
├── demo/                choreographed flight demo
└── test/                physics/config validation tests
```
