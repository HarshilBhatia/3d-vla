import open3d  # DON'T DELETE THIS!
from multiprocessing import Process, Manager

from pyrep.const import RenderMode

from rlbench import ObservationConfig
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.backend.utils import (
    task_file_to_task_class,
    float_array_to_rgb_image
)
import rlbench.backend.task as task

import os
import pickle
from PIL import Image
from rlbench.backend.const import *
import numpy as np
import random

from data.generation.customized_rlbench import CustomizedEnvironment

from absl import app
from absl import flags


MESH_POINT_FOLDER = 'mesh_points'
MESH_POINT_FORMAT = '%d.pkl'

FLAGS = flags.FLAGS

flags.DEFINE_string('save_path',
                    'data/train_dataset/microsteps/seed{seed}',
                    'Where to save the demos.')
flags.DEFINE_list('tasks', [],
                  'The tasks to collect. If empty, all tasks are collected.')
flags.DEFINE_list('image_size', [128, 128],
                  'The size of the images tp save.')
flags.DEFINE_enum('renderer',  'opengl3', ['opengl', 'opengl3'],
                  'The renderer to use. opengl does not include shadows, '
                  'but is faster.')
flags.DEFINE_integer('processes', 1,
                     'The number of parallel processes during collection.')
flags.DEFINE_integer('episodes_per_task', 10,
                     'The number of episodes to collect per task.')
flags.DEFINE_integer('variations', -1,
                     'Number of variations to collect per task. -1 for all.')
flags.DEFINE_integer('offset', 0,
                     'First variation id.')
flags.DEFINE_boolean('state', False,
                     'Record the state (not available for all tasks).')
flags.DEFINE_integer('seed', 0,
                     'Seed of randomness')
flags.DEFINE_integer('episode_start', 0,
                     'Starting episode index (for parallel jobs writing non-overlapping ranges).')


def check_and_make(dir):
    os.makedirs(dir, exist_ok=True)


class DemoSaver:

    def __init__(self, demo, example_path):
        self.demo = demo
        self.example_path = example_path

    def store(self, folder, attr):
        # Create folder
        path_ = os.path.join(self.example_path, folder)
        os.makedirs(path_, exist_ok=True)
        # Loop over demo and store
        for i, obs in enumerate(self.demo):
            # Read image
            img = obs.__getattribute__(attr)
            if 'rgb' in attr:
                img = Image.fromarray(img)
            elif 'depth' in attr:
                img = float_array_to_rgb_image(img, scale_factor=DEPTH_SCALE)
            elif 'mask' in attr:
                img = Image.fromarray((img * 255).astype(np.uint8))
            # Save image
            img.save(os.path.join(path_, IMAGE_FORMAT % i))
            # Set to None for pickling later
            obs.__setattr__(attr, None)


def save_demo(demo, example_path):
    ds = DemoSaver(demo, example_path)
    paths_attrs = [
        (LEFT_SHOULDER_RGB_FOLDER, 'left_shoulder_rgb'),
        (LEFT_SHOULDER_DEPTH_FOLDER, 'left_shoulder_depth'),
        (LEFT_SHOULDER_MASK_FOLDER, 'left_shoulder_mask'),
        (RIGHT_SHOULDER_RGB_FOLDER, 'right_shoulder_rgb'),
        (RIGHT_SHOULDER_DEPTH_FOLDER, 'right_shoulder_depth'),
        (RIGHT_SHOULDER_MASK_FOLDER, 'right_shoulder_mask'),
        (OVERHEAD_RGB_FOLDER, 'overhead_rgb'),
        (OVERHEAD_DEPTH_FOLDER, 'overhead_depth'),
        (OVERHEAD_MASK_FOLDER, 'overhead_mask'),
        (WRIST_RGB_FOLDER, 'wrist_rgb'),
        (WRIST_DEPTH_FOLDER, 'wrist_depth'),
        (WRIST_MASK_FOLDER, 'wrist_mask'),
        (FRONT_RGB_FOLDER, 'front_rgb'),
        (FRONT_DEPTH_FOLDER, 'front_depth'),
        (FRONT_MASK_FOLDER, 'front_mask')
    ]
    # Save image data first and then None them
    for folder, attr in paths_attrs:
        ds.store(folder, attr)

    # Store object point clouds
    mesh_point_path = os.path.join(example_path, MESH_POINT_FOLDER)
    os.makedirs(mesh_point_path, exist_ok=True)
    for i, obs in enumerate(demo):
        mesh_points = obs.mesh_points
        with open(os.path.join(mesh_point_path, MESH_POINT_FORMAT % i), 'wb') as f:
            pickle.dump(mesh_points, f)
        obs.__delattr__('mesh_points')

    # Save the low-dimension data
    with open(os.path.join(example_path, LOW_DIM_PICKLE), 'wb') as f:
        pickle.dump(demo, f)


def run(i, lock, task_index, results, file_lock, tasks):
    """Collect episodes_per_task total episodes per task, cycling variations round-robin.

    Episode ep_idx maps to:
      variation      = ep_idx % n_variations
      within-var idx = ep_idx // n_variations

    Parallel jobs use --episode_start to write non-overlapping episode ranges.
    """
    np.random.seed(FLAGS.seed + i)
    random.seed(FLAGS.seed + i)
    num_tasks = len(tasks)

    img_size = list(map(int, FLAGS.image_size))

    obs_config = ObservationConfig()
    obs_config.set_all(True)
    obs_config.right_shoulder_camera.image_size = img_size
    obs_config.left_shoulder_camera.image_size = img_size
    obs_config.overhead_camera.image_size = img_size
    obs_config.wrist_camera.image_size = img_size
    obs_config.front_camera.image_size = img_size

    # Store depth as 0 - 1
    obs_config.right_shoulder_camera.depth_in_meters = False
    obs_config.left_shoulder_camera.depth_in_meters = False
    obs_config.overhead_camera.depth_in_meters = False
    obs_config.wrist_camera.depth_in_meters = False
    obs_config.front_camera.depth_in_meters = False

    # We want to save the masks as rgb encodings.
    obs_config.left_shoulder_camera.masks_as_one_channel = False
    obs_config.right_shoulder_camera.masks_as_one_channel = False
    obs_config.overhead_camera.masks_as_one_channel = False
    obs_config.wrist_camera.masks_as_one_channel = False
    obs_config.front_camera.masks_as_one_channel = False

    # No need to save point cloud, we'll unproject them from depth
    obs_config.left_shoulder_camera.point_cloud = False
    obs_config.right_shoulder_camera.point_cloud = False
    obs_config.overhead_camera.point_cloud = False
    obs_config.wrist_camera.point_cloud = False
    obs_config.front_camera.point_cloud = False

    if FLAGS.renderer == 'opengl':
        obs_config.right_shoulder_camera.render_mode = RenderMode.OPENGL
        obs_config.left_shoulder_camera.render_mode = RenderMode.OPENGL
        obs_config.overhead_camera.render_mode = RenderMode.OPENGL
        obs_config.wrist_camera.render_mode = RenderMode.OPENGL
        obs_config.front_camera.render_mode = RenderMode.OPENGL

    rlbench_env = CustomizedEnvironment(
        action_mode=MoveArmThenGripper(JointVelocity(), Discrete()),
        obs_config=obs_config,
        headless=True
    )
    rlbench_env.launch()
    tasks_with_problems = results[i] = ''

    while True:
        with lock:
            if task_index.value >= num_tasks:
                print('Process', i, 'finished')
                break
            t = tasks[task_index.value]
            task_index.value += 1

        task_env = rlbench_env.get_task(t)
        n_variations = task_env.variation_count()

        ep_start = FLAGS.episode_start
        ep_end = ep_start + FLAGS.episodes_per_task
        descriptions_saved = set()

        for ep_idx in range(ep_start, ep_end):
            variation = ep_idx % n_variations
            within_var_idx = ep_idx // n_variations

            variation_path = os.path.join(
                FLAGS.save_path, task_env.get_name(),
                VARIATIONS_FOLDER % variation
            )
            check_and_make(variation_path)
            episodes_path = os.path.join(variation_path, EPISODES_FOLDER)
            check_and_make(episodes_path)
            episode_path = os.path.join(episodes_path, EPISODE_FOLDER % within_var_idx)

            if os.path.exists(episode_path):
                print('[SKIP] ep_idx=%d (var=%d within=%d) already exists' % (
                    ep_idx, variation, within_var_idx))
                continue

            task_env.set_variation(variation)
            descriptions, obs = task_env.reset()

            if variation not in descriptions_saved:
                with file_lock:
                    desc_path = os.path.join(variation_path, VARIATION_DESCRIPTIONS)
                    if not os.path.exists(desc_path):
                        with open(desc_path, 'wb') as f:
                            pickle.dump(descriptions, f)
                descriptions_saved.add(variation)

            print('Process %d // Task: %s // Variation: %d // Demo: %d' % (
                i, task_env.get_name(), variation, within_var_idx))

            attempts = 10
            while attempts > 0:
                try:
                    demo, = task_env.get_demos(amount=1, live_demos=True)
                except Exception as e:
                    attempts -= 1
                    if attempts > 0:
                        print('Process %d retrying ep_idx=%d: %s' % (i, ep_idx, e))
                        continue
                    problem = (
                        'Process %d failed task %s ep_idx=%d (var=%d within=%d): %s\n' % (
                            i, task_env.get_name(), ep_idx, variation, within_var_idx, e)
                    )
                    print(problem)
                    tasks_with_problems += problem
                    break
                with file_lock:
                    save_demo(demo, episode_path)
                break

    results[i] = tasks_with_problems
    rlbench_env.shutdown()


def main(argv):

    FLAGS.save_path = FLAGS.save_path.format(seed=FLAGS.seed)

    task_files = [t.replace('.py', '') for t in os.listdir(task.TASKS_PATH)
                  if t != '__init__.py' and t.endswith('.py')]

    if len(FLAGS.tasks) > 0:
        for t in FLAGS.tasks:
            if t not in task_files:
                raise ValueError('Task %s not recognised!.' % t)
        task_files = FLAGS.tasks

    tasks = [task_file_to_task_class(t) for t in task_files]

    manager = Manager()

    result_dict = manager.dict()
    file_lock = manager.Lock()

    task_index = manager.Value('i', 0)
    lock = manager.Lock()

    check_and_make(FLAGS.save_path)

    processes = [Process(
        target=run, args=(
            i, lock, task_index, result_dict, file_lock,
            tasks))
        for i in range(FLAGS.processes)]

    [t.start() for t in processes]
    [t.join() for t in processes]

    print('Data collection done!')
    for i in range(FLAGS.processes):
        print(result_dict[i])


if __name__ == '__main__':
  app.run(main)
