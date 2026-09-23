"""
Environment-sampling utilities for the numpy oracle (app.environmental.base_drone_env).

Used to be "PyBullet-specific glue code" (spawn_drone, building a PyBullet
multi-body for GUI visualization) -- removed: Isaac Sim is the only
visualization/simulation path now (app/environmental/base_drone_env_isaac.py),
so base_drone_env.py's render_mode="human" PyBullet path was removed too (see
that file). sample_wind_conditions has no PyBullet dependency and stays.
"""


def sample_wind_conditions(np_rand):
    rng=np_rand
    wind_vector = rng.uniform(-2.0, 2.0, size=3)
    mass_scale = rng.uniform(0.92, 1.08)
    return wind_vector, mass_scale

