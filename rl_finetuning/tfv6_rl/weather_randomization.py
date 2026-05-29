"""Bench2Drive-style weather randomization for residual RL training.

Training route files ship with empty ``<weathers/>`` blocks, so CARLA falls back to a
single default (clear noon) weather. The bench2drive evaluation, by contrast, spans 16
distinct presets (clear/cloudy/rain/fog across day, sunset and night). This module draws
weather from a distribution matched to those presets so the residual policy is exposed to
the same lighting and precipitation variety it is benchmarked on.

Enabled via the ``randomize_weather`` config flag; see ``EnvAgentTFv6.agent_route_init``.
"""

import carla
import numpy as np

# The 16 weather presets used by data/benchmark_routes/bench2drive20LT.xml.
# Fields: (cloudiness, precipitation, precipitation_deposits, wetness,
#          fog_density, wind_intensity, sun_altitude_angle, sun_azimuth_angle).
# Sampled uniformly (not by the benchmark's empirical frequency) so adverse
# conditions get balanced training exposure rather than being dominated by clear noon.
_BENCH2DRIVE_PRESETS = (
    (5, 0, 0, 0, 2, 10, 90, -1),
    (40, 0, 50, 0, 2, 10, 15, -1),
    (5, 0, 50, 0, 10, 10, 15, -1),
    (80, 60, 60, 80, 60, 60, -90, -1),
    (80, 0, 50, 20, 3, 10, 45, -1),
    (100, 60, 60, 0, 50, 60, 45, -1),
    (100, 50, 100, 80, 10, 80, 45, -1),
    (100, 100, 90, 0, 7, 100, 45, -1),
    (100, 100, 50, 50, 10, 100, 45, -1),
    (5, 0, 0, 0, 2, 10, 15, -1),
    (5, 0, 0, 0, 50, 10, 45, -1),
    (5, 0, 0, 0, 50, 10, 15, -1),
    (100, 100, 90, 100, 100, 100, -90, -1),
    (5, 0, 0, 0, 60, 10, -90, -1),
    (60, 60, 60, 0, 3, 60, 45, -1),
    (20, 30, 50, 0, 3, 30, 45, -1),
)

# Maximum absolute per-parameter jitter added on top of the chosen preset.
_JITTER = 10.0


def sample_weather(rng: np.random.Generator) -> carla.WeatherParameters:
    """Return ``carla.WeatherParameters`` drawn from a bench2drive-like distribution.

    A preset is chosen uniformly at random and each parameter is perturbed by a small
    bounded jitter, so repeated runs of the same route never receive identical weather
    while staying in a realistic neighborhood of a real preset (preserving the natural
    correlations between cloud/rain/wetness/fog).

    Args:
        rng: A NumPy ``Generator``. Pass a per-process, unseeded generator so parallel
            workers and route repetitions diverge.
    """
    cloud, precip, deposits, wetness, fog, wind, sun_alt, sun_azi = (
        _BENCH2DRIVE_PRESETS[rng.integers(len(_BENCH2DRIVE_PRESETS))]
    )

    def jitter(value, lo=0.0, hi=100.0):
        return float(np.clip(value + rng.uniform(-_JITTER, _JITTER), lo, hi))

    return carla.WeatherParameters(
        cloudiness=jitter(cloud),
        precipitation=jitter(precip),
        precipitation_deposits=jitter(deposits),
        wetness=jitter(wetness),
        fog_density=jitter(fog),
        wind_intensity=jitter(wind),
        sun_altitude_angle=jitter(sun_alt, lo=-90.0, hi=90.0),
        sun_azimuth_angle=float(sun_azi),
    )
