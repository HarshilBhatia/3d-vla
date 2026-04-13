from scipy.spatial.transform import Rotation as R

from pyrep.objects.shape import Shape
from rlbench.environment import Environment
from rlbench.backend.scene import Scene
from rlbench.sim2real.domain_randomization_scene import DomainRandomizationScene


class CustomizedScene(Scene):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Cache of object handle -> (name, local-frame vertices)
        # Mesh geometry never changes; only poses change per step.
        self._mesh_cache = {}   # handle -> (name_or_key, vertices ndarray)
        self._robot_static = None  # list of (key, obj, vertices) for arm + table
        self._task_static = None   # list of (key, obj, vertices) per task reset

    def _vertices(self, obj):
        """Return cached local-frame vertices for obj (IPC call only on first use)."""
        handle = obj.get_handle()
        if handle not in self._mesh_cache:
            verts, _, _ = obj.get_mesh_data()
            self._mesh_cache[handle] = verts
        return self._mesh_cache[handle]

    def _build_robot_static(self):
        """Build the static arm-joint + table entry list (once per scene lifetime)."""
        joints = self.robot.arm.get_visuals()
        joints.append(Shape('diningTable_visible'))
        self._robot_static = [
            (obj.get_name(), obj, self._vertices(obj)) for obj in joints
        ]

    def _build_task_static(self):
        """Build task-object entry list (once per task reset, not per step)."""
        self._task_static = [
            (
                obj.get_name() + "_" + str(obj.get_color()),
                obj,
                self._vertices(obj),
            )
            for (obj, _) in self.task._initial_objs_in_scene
            if isinstance(obj, Shape)
        ]

    def reset(self):
        super().reset()
        # _initial_objs_in_scene is repopulated by reset(); invalidate task cache.
        self._task_static = None

    def get_observation(self):
        obs = super().get_observation()

        # Build caches lazily
        if self._robot_static is None:
            self._build_robot_static()
        if self._task_static is None:
            self._build_task_static()

        mesh_points = {}

        for key, obj, verts in self._robot_static:
            pose = obj.get_pose()
            rot = R.from_quat(pose[3:7]).as_matrix()
            mesh_points[key] = (rot @ verts.T).T + pose[:3].reshape(1, 3)

        for key, obj, verts in self._task_static:
            pose = obj.get_pose()
            rot = R.from_quat(pose[3:7]).as_matrix()
            mesh_points[key] = (rot @ verts.T).T + pose[:3].reshape(1, 3)

        obs.mesh_points = mesh_points
        obs = self.task.decorate_observation(obs)
        return obs


class CustomizedDomainRandomizationScene(CustomizedScene, DomainRandomizationScene):
    pass


class CustomizedEnvironment(Environment):

    def launch(self):
        super().launch()
        if self._randomize_every is None:
            self._scene = CustomizedScene(
                self._pyrep, self._robot, self._obs_config, self._robot_setup)
        else:
            self._scene = CustomizedDomainRandomizationScene(
                self._pyrep, self._robot, self._obs_config, self._robot_setup,
                self._randomize_every, self._frequency,
                self._visual_randomization_config,
                self._dynamics_randomization_config)
