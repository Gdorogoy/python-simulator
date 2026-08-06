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
- **guidance** - the learning block: an actor-critic network trained with PPO so the interceptor learns to reach its target
- **navigation** - reads the true state from dynamics (read-only, never writes to it), adds sensor noise/delay/drift, and outputs an estimated state; a one-way transform, not part of the physics update
- **demo** - runs a scripted flight plan (altitude changes, tilts, translation, circling, yaw spin) through the physics engine to showcase what the drone can do
- **test** - checks that a given drone configuration is physically valid (mixer invertibility, hover equilibrium, stability, etc.)
- **main.py** - entry point; menu to run the demo, the tests, or the RL training/interceptor

## Folder structure
```
app/
├── main.py             entry point (demo / test / RL menu)
├── dynamics/            physics engine (drone structure + force calculations)
├── environmental/       PyBullet glue + Gymnasium interceptor environment
├── guidance/            PPO training (actor-critic)
├── navigation/          read-only state estimator (sensor noise/delay/drift)
├── demo/                choreographed flight demo
└── test/                physics/config validation tests
```
