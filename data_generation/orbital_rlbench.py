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

from data_generation.customized_rlbench import CustomizedScene, CustomizedEnvironment


class OrbitalScene(CustomizedScene):
    """CustomizedScene that also captures two orbital camera sensors."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._orbital_left = None
        self._orbital_right = None

    def set_orbital_sensors(self, left, right):
        """Register the two orbital VisionSensors to capture at each step."""
        self._orbital_left = left
        self._orbital_right = right

    def clear_orbital_sensors(self):
        self._orbital_left = None
        self._orbital_right = None

    def get_observation(self):
        obs = super().get_observation()

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

        return obs


class OrbitalEnvironment(CustomizedEnvironment):
    """CustomizedEnvironment that uses OrbitalScene instead of CustomizedScene."""

    def launch(self):
        # Call the grandparent (Environment.launch) then replace _scene
        from rlbench.environment import Environment
        Environment.launch(self)
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
