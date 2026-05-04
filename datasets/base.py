import json

import torch
from torch.utils.data import Dataset

from .utils import to_tensor, read_zarr_with_cache, to_relative_action


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
        mem_limit=8,  # cache limit per dataset class in GigaBytes
        actions_only=False,  # return actions without observations
        chunk_size=4,  # chunk size for zarr
        num_history=1,  # number of visual history frames (1 = current frame only)
    ):
        super().__init__()
        self.copies = self.train_copies if copies is None else copies
        self._relative_action = relative_action
        self._actions_only = actions_only
        self.chunk_size = chunk_size
        self.num_history = num_history

        # Load instructions
        self._instructions = self._load_instructions(instructions)

        # base tasks -- can be overwritten 
        self.tasks = list(self._instructions.keys())


        # Load all annotations lazily
        self.annos = read_zarr_with_cache(root, mem_gb=mem_limit)

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

    def _get_single_frame_hist(self, zarr_idx, attr, filter_cam=False):
        """Return (num_history, *shape) for one zarr index with demo_id boundary padding."""
        demo_curr = int(self.annos['demo_id'][zarr_idx])
        frames = []
        first_valid = None
        for k in range(self.num_history - 1, -1, -1):  # oldest → current
            j = zarr_idx - k
            if j >= 0 and int(self.annos['demo_id'][j]) == demo_curr:
                t = to_tensor(self.annos[attr][j:j + 1])[0]
                if filter_cam and self.camera_inds is not None:
                    t = t[self.camera_inds]
                if first_valid is None:
                    first_valid = t
                frames.append(t)
            else:
                frames.append(None)
        fallback = first_valid if first_valid is not None else (
            to_tensor(self.annos[attr][zarr_idx:zarr_idx + 1])[0]
        )
        return torch.stack([f if f is not None else fallback for f in frames])

    def _get_attr_hist(self, idx, attr, filter_cam=False):
        """Return (chunk_size, num_history, *shape) — each sample with nhist history frames."""
        return torch.stack([
            self._get_single_frame_hist(idx + i, attr, filter_cam)
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
            "proprioception": self._get_proprioception(idx),  # tensor(1, 8)
            "action": self._get_action(idx)  # tensor(T, 8)
        }

    def __len__(self):
        return self.copies * (len(self.annos['action']) // self.chunk_size)
