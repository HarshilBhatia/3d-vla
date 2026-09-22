import json
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from .utils import to_tensor, read_zarr_with_cache, load_zarr_to_ram, to_relative_action


class BaseDataset(Dataset):
    """Base dataset."""
    camera_inds = None
    quat_format = 'xyzw'

    def __init__(
        self,
        root,  # the directory path of the dataset
        instructions,  # path to instruction file
        copies=None,  # copy the dataset for less loader restarts
        relative_action=False,  # whether to return relative actions
        mem_limit=8,  # cache limit per dataset class in GigaBytes (ignored when preload=True)
        actions_only=False,  # return actions without observations
        chunk_size=4,  # chunk size for zarr
        visual_num_history=1,  # number of visual history frames (1 = current frame only)
        proprio_num_history=None,
        preload=False,  # load entire dataset into RAM at init
        visual_history_sampling='consecutive',  # consecutive | random
    ):
        super().__init__()
        self.copies = self.train_copies if copies is None else copies
        self._relative_action = relative_action
        self._actions_only = actions_only
        self.chunk_size = chunk_size
        self.visual_num_history = visual_num_history
        self.proprio_num_history = visual_num_history if proprio_num_history is None else proprio_num_history
        if visual_history_sampling not in ('consecutive', 'random'):
            raise ValueError(f"visual_history_sampling must be consecutive or random, got {visual_history_sampling!r}")
        self.visual_history_sampling = visual_history_sampling
        self._demo_ids = None

        # Load instructions
        self._instructions = self._load_instructions(instructions)

        # base tasks -- can be overwritten 
        self.tasks = list(self._instructions.keys())


        # Load all annotations — either lazily (LRU cache) or fully into RAM
        self.annos = read_zarr_with_cache(root, mem_gb=mem_limit)
        if preload:
            print(f"Preloading dataset to RAM: {root}")
            self.annos = load_zarr_to_ram(self.annos)
            print("Preload complete.")

        # Sanity check
        len_ = len(self.annos['action'])
        for key in self.annos:
            assert len(self.annos[key]) == len_, f'length mismatch in {key}'

    def _load_instructions(self, instruction_file):
        return json.load(open(instruction_file))

    def _normalize_idx(self, idx):
        idx = idx % (len(self.annos['action']) // self.chunk_size)
        return idx * self.chunk_size

    def _get_attr_by_idx(self, idx, attr, filter_cam=False):
        t = to_tensor(self.annos[attr][idx:idx + self.chunk_size])
        if filter_cam and self.camera_inds is not None:
            t = t[:, self.camera_inds]
        return t

    def _get_task(self, idx):
        return ["task"] * self.chunk_size

    def _get_instr(self, idx):
        return ["instruction"] * self.chunk_size

    def _get_rgb(self, idx, key='rgb'):
        return self._get_attr_by_idx(idx, key, True)

    def _consecutive_hist_indices(self, zarr_idx):
        """Oldest → current zarr indices; slots before the demo start repeat its first frame."""
        demo_curr = int(self.annos['demo_id'][zarr_idx])
        valid = [
            zarr_idx - k if zarr_idx - k >= 0 and int(self.annos['demo_id'][zarr_idx - k]) == demo_curr else None
            for k in range(self.visual_num_history - 1, -1, -1)
        ]
        fallback = next(j for j in valid if j is not None)
        return [fallback if j is None else j for j in valid]

    def _random_hist_indices(self, zarr_idx):
        """K-1 distinct random past indices of the same demo, current index last."""
        if self._demo_ids is None:
            self._demo_ids = np.asarray(self.annos['demo_id'][:])
        start = zarr_idx
        while start > 0 and self._demo_ids[start - 1] == self._demo_ids[zarr_idx]:
            start -= 1
        past = list(range(start, zarr_idx))
        picked = random.sample(past, min(len(past), self.visual_num_history - 1))
        pad = self.visual_num_history - 1 - len(picked)
        return [start] * pad + picked + [zarr_idx]

    def _get_hist_indices(self, idx):
        """Per-sample source indices for a chunk, shared by every visual attribute."""
        fn = self._random_hist_indices if self.visual_history_sampling == 'random' else self._consecutive_hist_indices
        return [fn(idx + i) for i in range(self.chunk_size)]

    def _get_single_frame_hist(self, indices, attr, filter_cam=False):
        """Return (visual_num_history, *shape) gathered at the given zarr indices."""
        frames = []
        for j in indices:
            t = to_tensor(self.annos[attr][j:j + 1])[0]
            if filter_cam and self.camera_inds is not None:
                t = t[self.camera_inds]
            frames.append(t)
        return torch.stack(frames)

    def _get_attr_hist(self, idx, attr, filter_cam=False, hist_indices=None):
        """Return (chunk_size, visual_num_history, *shape) — each sample with nhist history frames."""
        if hist_indices is None:
            hist_indices = self._get_hist_indices(idx)
        return torch.stack([
            self._get_single_frame_hist(indices, attr, filter_cam)
            for indices in hist_indices
        ])

    def _get_single_proprio_hist(self, zarr_idx):
        """Return causal proprio history as ``(visual_num_history, nhand, state)``.

        Each Zarr row stores ``[s_(t-2), s_(t-1), s_t]``.  For a longer causal
        window, assemble the final/current slot from preceding records.
        Keep the same past-to-current layout and earliest-state left padding as
        visual history: ``[s_(t-K+1), ..., s_t]``.
        """
        demo_curr = int(self.annos['demo_id'][zarr_idx])
        states = []
        fallback = None
        for k in range(self.visual_num_history - 1, -1, -1):  # past → current
            j = zarr_idx - k
            if j >= 0 and int(self.annos['demo_id'][j]) == demo_curr:
                # Per-record final slot is the state at this record's timestamp.
                state = to_tensor(self.annos['proprioception'][j:j + 1])[0, -1]
                if fallback is None:
                    fallback = state
                states.append(state)
            else:
                states.append(None)
        if fallback is None:
            fallback = to_tensor(self.annos['proprioception'][zarr_idx:zarr_idx + 1])[0, 0]
        return torch.stack([state if state is not None else fallback for state in states])

    def _get_proprioception_hist(self, idx):
        return torch.stack([
            self._get_single_proprio_hist(idx + i)
            for i in range(self.chunk_size)
        ])

    def _get_depth(self, idx, key='depth'):
        return self._get_attr_by_idx(idx, key, True)

    def _get_proprioception(self, idx):
        return self._get_attr_by_idx(idx, 'proprioception', False)

    def _get_action(self, idx):
        if self._relative_action:
            if 'rel_action' in self.annos:
                return self._get_attr_by_idx(idx, 'rel_action', False)
            else:
                action = self._get_attr_by_idx(idx, 'action', False)
                prop = self._get_proprioception(idx)[[-1]]
                action = to_relative_action(action, prop, self.quat_format)
        else:
            action = self._get_attr_by_idx(idx, 'action', False)
        return action

    def __getitem__(self, idx):
        """
        self.annos: {
            action: (N, T, 8) float
            depth: (N, n_cam, H, W) float16
            proprioception: (N, nhist, 8) float
            rgb: (N, n_cam, 3, H, W) uint8
        }
        In addition self.annos may contain fields for task/instruction ids
        """
        idx = self._normalize_idx(idx)
        
        if self._actions_only:
            return {"action": self._get_action(idx)}
        return {
            "task": self._get_task(idx),
            "instr": self._get_instr(idx),  # [str]
            "rgb": self._get_rgb(idx),  # tensor(n_cam, 3, H, W)
            "depth": self._get_depth(idx),  # tensor(n_cam, H, W)
            "proprioception": self._get_proprioception_hist(idx)
            if self.visual_num_history > 1 and 'demo_id' in self.annos
            else self._get_proprioception(idx),
            "action": self._get_action(idx)  # tensor(T, 8)
        }

    def __len__(self):
        return self.copies * (len(self.annos['action']) // self.chunk_size)
