"""
OrbitalScene / OrbitalEnvironment — extends CustomizedScene to capture
RGB + depth from two externally placed orbital VisionSensors at every step.

Usage:
    scene.set_orbital_sensors(left_sensor, right_sensor)
    # Then call task_env.get_demos(...) as normal.
    # Each obs will have:
    #   obs.orbital_left_rgb    (H, W, 3) uint8
    #   obs.orbital_left_depth  (H, W)    float32 (metres)
    #   obs.orbital_right_rgb   (H, W, 3) uint8
    #   obs.orbital_right_depth (H, W)    float32 (metres)
"""

import time as _time

from data_generation.customized_rlbench import CustomizedScene, CustomizedEnvironment


class OrbitalScene(CustomizedScene):
    """CustomizedScene that also captures two orbital camera sensors."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._orbital_left = None
        self._orbital_right = None
        # Per-demo step timing accumulators (reset via reset_step_timers())
        self._t_obs_wrist   = 0.0   # wrist + standard cameras (super().get_observation)
        self._t_obs_orbital = 0.0   # orbital left + right sensor captures
        self._t_physics     = 0.0   # pyrep.step() + task.step()
        self._n_steps       = 0

    # ------------------------------------------------------------------
    # Timer helpers
    # ------------------------------------------------------------------

    def reset_step_timers(self):
        """Call before get_demos() to start fresh per-step timing."""
        self._t_obs_wrist   = 0.0
        self._t_obs_orbital = 0.0
        self._t_physics     = 0.0
        self._n_steps       = 0

    def get_step_timers(self):
        """Return accumulated timing dict after get_demos() completes."""
        return dict(
            obs_wrist=self._t_obs_wrist,
            obs_orbital=self._t_obs_orbital,
            physics=self._t_physics,
            n_steps=self._n_steps,
        )

    # ------------------------------------------------------------------
    # Timed overrides
    # ------------------------------------------------------------------

    def step(self):
        """Advance physics — timed separately from observation capture."""
        t0 = _time.perf_counter()
        self.pyrep.step()
        self.task.step()
        if self._step_callback is not None:
            self._step_callback()
        self._t_physics += _time.perf_counter() - t0
        self._n_steps += 1

    def set_orbital_sensors(self, left, right):
        """Register the two orbital VisionSensors to capture at each step."""
        self._orbital_left = left
        self._orbital_right = right

    def clear_orbital_sensors(self):
        self._orbital_left = None
        self._orbital_right = None

    def get_observation(self):
        t0 = _time.perf_counter()
        obs = super().get_observation()
        self._t_obs_wrist += _time.perf_counter() - t0

        t1 = _time.perf_counter()
        if self._orbital_left is not None:
            self._orbital_left.handle_explicitly()
            obs.orbital_left_rgb = (
                self._orbital_left.capture_rgb() * 255
            ).clip(0, 255).astype("uint8")
            obs.orbital_left_depth = self._orbital_left.capture_depth(
                in_meters=True
            )
        else:
            obs.orbital_left_rgb = None
            obs.orbital_left_depth = None

        if self._orbital_right is not None:
            self._orbital_right.handle_explicitly()
            obs.orbital_right_rgb = (
                self._orbital_right.capture_rgb() * 255
            ).clip(0, 255).astype("uint8")
            obs.orbital_right_depth = self._orbital_right.capture_depth(
                in_meters=True
            )
        else:
            obs.orbital_right_rgb = None
            obs.orbital_right_depth = None
        self._t_obs_orbital += _time.perf_counter() - t1

        return obs


class OrbitalEnvironment(CustomizedEnvironment):
    """CustomizedEnvironment that uses OrbitalScene instead of CustomizedScene."""

    def launch(self):
        # Call the grandparent (Environment.launch) then replace _scene.
        # Environment.launch creates an intermediate Scene whose
        # _set_camera_properties() permanently removes any camera whose
        # channels are all disabled.  Swap in a full ObservationConfig so that
        # no cameras are deleted, then restore the real config for OrbitalScene.
        from rlbench.environment import Environment
        from rlbench.observation_config import ObservationConfig
        real_obs_config = self._obs_config
        self._obs_config = ObservationConfig()  # all cameras on — nothing removed
        Environment.launch(self)
        self._obs_config = real_obs_config
        if self._randomize_every is None:
            self._scene = OrbitalScene(
                self._pyrep, self._robot, self._obs_config, self._robot_setup
            )
        else:
            # Domain-randomization variant — inherit domain-rand mixin too
            from rlbench.sim2real.domain_randomization_scene import (
                DomainRandomizationScene,
            )

            class OrbitalDomainRandomizationScene(
                OrbitalScene, DomainRandomizationScene
            ):
                pass

            self._scene = OrbitalDomainRandomizationScene(
                self._pyrep,
                self._robot,
                self._obs_config,
                self._robot_setup,
                self._randomize_every,
                self._frequency,
                self._visual_randomization_config,
                self._dynamics_randomization_config,
            )
