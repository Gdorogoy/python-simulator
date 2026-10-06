"""Ambient sampling for the numpy env (wind, mass scale). Currently not called: see docs.md."""


def sample_wind_conditions(np_rand):
    rng=np_rand
    wind_vector = rng.uniform(-2.0, 2.0, size=3)
    mass_scale = rng.uniform(0.92, 1.08)
    return wind_vector, mass_scale

