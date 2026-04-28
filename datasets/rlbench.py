import json
import random
import numpy as np

from .base import BaseDataset


PERACT_TASKS = [
    "place_cups", "close_jar", "insert_onto_square_peg",
    "light_bulb_in", "meat_off_grill", "open_drawer",
    "place_shape_in_shape_sorter", "place_wine_at_rack_location",
    "push_buttons", "put_groceries_in_cupboard",
    "put_item_in_drawer", "put_money_in_safe", "reach_and_drag",
    "slide_block_to_color_target", "stack_blocks", "stack_cups",
    "sweep_to_dustpan_of_size", "turn_tap"
]
PERACT2_TASKS = [
    'bimanual_push_box',
    'bimanual_lift_ball',
    'bimanual_dual_push_buttons',
    'bimanual_pick_plate',
    'bimanual_put_item_in_drawer',
    'bimanual_put_bottle_in_fridge',
    'bimanual_handover_item',
    'bimanual_pick_laptop',
    'bimanual_straighten_rope',
    'bimanual_sweep_to_dustpan',
    'bimanual_lift_tray',
    'bimanual_handover_item_easy',
    'bimanual_take_tray_out_of_oven'
]


class RLBenchDataset(BaseDataset):
    """RLBench dataset."""
    quat_format= 'xyzw'

    def __init__(
        self,
        root,
        instructions,
        copies=None,
        relative_action=False,
        mem_limit=8,
        actions_only=False,
        chunk_size=4,
        num_history=1,
        filter_tasks=None  # List of task names to include, None means all tasks
    ):
        super().__init__(
            root=root,
            instructions=instructions,
            copies=copies,
            relative_action=relative_action,
            mem_limit=mem_limit,
            actions_only=actions_only,
            chunk_size=chunk_size,
            num_history=num_history,
        )
        
        # Store filter_tasks for later use
        self.filter_tasks = filter_tasks
        
        # Filter by tasks if specified
        if filter_tasks is not None:
            self._filter_by_tasks(filter_tasks)

    def _filter_by_tasks(self, filter_tasks):
        """Filter dataset to only include samples from specified tasks."""
        if not isinstance(filter_tasks, list):
            filter_tasks = [filter_tasks]
        
        # Convert task names to task IDs
        task_ids_to_keep = [self.tasks.index(task) for task in filter_tasks if task in self.tasks]
        
        if not task_ids_to_keep:
            raise ValueError(f"None of the specified tasks {filter_tasks} found in task list {self.tasks}")
        
        # Create mask for samples to keep
        task_ids = np.array(self.annos['task_id'][:])  # Load into numpy array
        unique_task_ids = np.unique(task_ids)
        
        # Handle single-task zarr files: if all samples have the same task_id and we're filtering
        # for a single task, assume the zarr file contains only that task (regardless of task_id value)
        if len(unique_task_ids) == 1 and len(filter_tasks) == 1:
            # Single task zarr file - remap task_id to the correct global task index
            requested_task = filter_tasks[0]
            if requested_task in self.tasks:
                correct_task_id = self.tasks.index(requested_task)
                # Convert all zarr arrays to numpy arrays (since zarr group is read-only)
                # and remap task_id to the correct global index
                num_samples = len(task_ids)
                converted_annos = {}
                for key in self.annos:
                    if hasattr(self.annos[key], '__getitem__'):
                        # Load zarr array into numpy
                        converted_annos[key] = np.array(self.annos[key][:])
                    else:
                        converted_annos[key] = self.annos[key]
                
                # Remap task_id to the correct global index
                converted_annos['task_id'] = np.full(num_samples, correct_task_id, dtype=np.uint8)
                
                # Replace the zarr group with numpy arrays
                self.annos = converted_annos
                
                print(f"Single-task zarr detected (all samples have task_id={unique_task_ids[0]}). "
                      f"Remapped to task_id={correct_task_id} for task: {requested_task}. "
                      f"Keeping all {num_samples} samples.")
                return
        
        # Multi-task zarr: use task IDs from the global task list
        unique_task_names = [self.tasks[int(tid)] for tid in unique_task_ids if int(tid) < len(self.tasks)]
        print(f"Debug: Found task IDs {unique_task_ids.tolist()} in dataset")
        print(f"Debug: Corresponding task names: {unique_task_names}")
        print(f"Debug: Looking for task IDs: {task_ids_to_keep} (tasks: {filter_tasks})")
        
        mask = np.isin(task_ids, task_ids_to_keep)
        
        # Convert boolean mask to integer indices (zarr doesn't support boolean indexing)
        indices = np.where(mask)[0]
        
        if len(indices) == 0:
            raise ValueError(
                f"No samples found for tasks {filter_tasks}. "
                f"Available tasks in dataset: {unique_task_names}. "
                f"Task IDs in dataset: {unique_task_ids.tolist()}, "
                f"Looking for IDs: {task_ids_to_keep}"
            )
        
        # Filter all annotation arrays - convert zarr arrays to numpy arrays first
        filtered_annos = {}
        for key in self.annos:
            if hasattr(self.annos[key], '__getitem__'):
                # Load zarr array into numpy and filter
                arr = np.array(self.annos[key][:])
                filtered_annos[key] = arr[indices]
            else:
                filtered_annos[key] = self.annos[key]
        
        # Replace annotations with filtered versions
        self.annos = filtered_annos
        
        # Verify lengths match
        len_ = len(self.annos['action'])
        for key in self.annos:
            assert len(self.annos[key]) == len_, f'length mismatch in {key} after filtering'
        
        print(f"Filtered to {len(self.annos['action'])} samples from tasks: {filter_tasks}")
    
    def _get_task(self, idx):
        return [
            self.tasks[int(tid)]
            for tid in self.annos['task_id'][idx:idx + self.chunk_size]
        ]

    def _get_instr(self, idx):
        return [
            random.choice(self._instructions[self.tasks[int(t)]][str(int(v))])
            for t, v in zip(
                self.annos['task_id'][idx:idx + self.chunk_size],
                self.annos['variation'][idx:idx + self.chunk_size]
            )
        ]

    def _get_rgb2d(self, idx):
        if self.camera_inds2d is not None:
            return self._get_attr_by_idx(idx, 'rgb', False)[:, self.camera_inds2d]
        return None

    def _get_extrinsics(self, idx):
        return self._get_attr_by_idx(idx, 'extrinsics', True)

    def _get_intrinsics(self, idx):
        return self._get_attr_by_idx(idx, 'intrinsics', True)

    def __getitem__(self, idx):
        """
        self.annos: {
            action: (N, T, 8) float
            depth: (N, n_cam, H, W) float16
            proprioception: (N, nhist, 8) float
            rgb: (N, n_cam, 3, H, W) uint8
            task_id: (N,) uint8
            variation: (N,) uint8
            extrinsics: (N, n_cam, 4, 4) float
            intrinsics: (N, n_cam, 3, 3) float
        }
        """
        # First detect which copy we fall into
        idx = idx % (len(self.annos['action']) // self.chunk_size)
        # and then which chunk
        idx = idx * self.chunk_size
        if self._actions_only:
            return {"action": self._get_action(idx)}
        use_hist = self.num_history > 1 and 'demo_id' in self.annos
        return {
            "task": self._get_task(idx),  # [str]
            "instr": self._get_instr(idx),  # [str]
            "rgb": self._get_attr_hist(idx, 'rgb', True) if use_hist else self._get_rgb(idx),
            "depth": self._get_attr_hist(idx, 'depth', True) if use_hist else self._get_depth(idx),
            "rgb2d": self._get_rgb2d(idx),  # tensor(n_cam2d, 3, H, W)
            "proprioception": self._get_proprioception(idx),  # tensor(1, 8)
            "action": self._get_action(idx),  # tensor(T, 8)
            "extrinsics": self._get_attr_hist(idx, 'extrinsics', True) if use_hist else self._get_extrinsics(idx),
            "intrinsics": self._get_attr_hist(idx, 'intrinsics', True) if use_hist else self._get_intrinsics(idx),
        }


class HiveformerDataset(RLBenchDataset):
    cameras = ("wrist", "front")
    camera_inds = None
    train_copies = 100
    camera_inds2d = None

    def _load_instructions(self, instruction_file):
        instr = json.load(open(instruction_file))
        self.tasks = list(instr.keys())
        return instr


class PeractDataset(RLBenchDataset):
    """RLBench dataset under Peract setup."""
    tasks = PERACT_TASKS
    cameras = ("left_shoulder", "right_shoulder", "wrist", "front")
    camera_inds = None
    train_copies = 10
    camera_inds2d = None

    def __getitem__(self, idx):
        """
        self.annos: {
            action: (N, T, 8) float
            depth: (N, n_cam, H, W) float16
            proprioception: (N, nhist, 8) float
            rgb: (N, n_cam, 3, H, W) uint8
            task_id: (N,) uint8
            variation: (N,) uint8
            extrinsics: (N, n_cam, 4, 4) float
            intrinsics: (N, n_cam, 3, 3) float
        }
        """
        # First detect which copy we fall into
        idx = idx % (len(self.annos['action']) // self.chunk_size)
        # and then which chunk
        idx = idx * self.chunk_size
        if self._actions_only:
            return {"action": self._get_action(idx)}
        use_hist = self.num_history > 1 and 'demo_id' in self.annos
        return {
            "task": self._get_task(idx),  # [str]
            "instr": self._get_instr(idx),  # [str]
            "rgb": self._get_attr_hist(idx, 'rgb', True) if use_hist else self._get_rgb(idx),
            "pcd": self._get_attr_hist(idx, 'pcd', True) if use_hist else self._get_attr_by_idx(idx, 'pcd', True),
            "proprioception": self._get_proprioception(idx),  # tensor(1, 8)
            "action": self._get_action(idx),  # tensor(T, 8)
        }


class PeractTwoCamDataset(PeractDataset):
    """RLBench dataset under Peract setup."""
    tasks = PERACT_TASKS
    cameras = ("wrist", "front")
    camera_inds = [2, 3]
    train_copies = 10
    camera_inds2d = None


class PeractCollectedDataset(RLBenchDataset):
    """Self-collected unimanual PerAct data (depth+extrinsics+intrinsics format)."""
    tasks = PERACT_TASKS
    cameras = ("left_shoulder", "right_shoulder", "wrist", "front")
    camera_inds = None
    train_copies = 1
    camera_inds2d = None


class Peract2Dataset(RLBenchDataset):
    """RLBench dataset under Peract2 setup."""
    tasks = PERACT2_TASKS
    cameras = ("front", "wrist_left", "wrist_right")
    camera_inds = None
    train_copies = 10
    camera_inds2d = None


class Peract2SingleCamDataset(RLBenchDataset):
    """RLBench dataset under Peract2 setup."""
    tasks = PERACT2_TASKS
    cameras = ("front",)
    camera_inds = (0,)  # use only front camera
    train_copies = 10
    camera_inds2d = None


class OrbitalWristDataset(RLBenchDataset):
    """RLBench dataset with orbital left/right + wrist cameras."""
    tasks = PERACT2_TASKS
    cameras = ("orbital_left", "orbital_right", "wrist")
    camera_inds = None
    train_copies = 10
    camera_inds2d = None

    def __getitem__(self, idx):
        idx = idx % (len(self.annos['action']) // self.chunk_size)
        idx = idx * self.chunk_size
        if self._actions_only:
            return {"action": self._get_action(idx)}
        use_hist = self.num_history > 1 and 'demo_id' in self.annos
        return {
            "task": self._get_task(idx),
            "instr": self._get_instr(idx),
            "rgb": self._get_attr_hist(idx, 'rgb', True) if use_hist else self._get_rgb(idx),
            "pcd": self._get_attr_hist(idx, 'pcd', True) if use_hist else self._get_attr_by_idx(idx, 'pcd', True),
            "proprioception": self._get_proprioception(idx),
            "action": self._get_action(idx),
        }
