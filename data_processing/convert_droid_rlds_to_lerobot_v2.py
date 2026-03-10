"""
Convert DROID RLDS to LeRobot v2 dataset format:

  dataset/
  ├── data/chunk-000/
  │   ├── episode_000000.parquet   (tabular only: state, action, indices, no images)
  │   └── ...
  ├── videos/chunk-000/
  │   └── observation.images.<camera_name>/
  │       ├── episode_000000.mp4
  │       └── ...
  └── meta/
      ├── info.json
      ├── tasks.jsonl
      ├── episodes.jsonl
      ├── stats.json
      └── episode_index_to_id.json (optional)
"""
import os

# Force CPU-only before any TensorFlow/CUDA code runs (required on CPU-only nodes, e.g. RM-shared).
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import wandb
except ImportError:
    wandb = None  # type: ignore
import pyarrow as pa
import pyarrow.parquet as pq

# DROID RLDS: 15 fps, 180x320 RGB
FPS = 15
IMAGE_SHAPE = (180, 320, 3)
EPISODES_PER_CHUNK = 1000
CHECKPOINT_INTERVAL = 100  # write meta/ snapshot every this many completed episodes
CAMERA_KEYS = [
    "observation.images.wrist_image_left",
    "observation.images.exterior_image_1_left",
    "observation.images.exterior_image_2_left",
]

# Global cache: choose AV1 encoder once per process.
_AV1_ENCODER: Optional[str] = None


def _pick_av1_encoder() -> str:
    """Return ffmpeg encoder name for AV1 (prefers libsvtav1)."""
    global _AV1_ENCODER
    if _AV1_ENCODER is not None:
        return _AV1_ENCODER
    import subprocess

    try:
        out = subprocess.check_output(["ffmpeg", "-hide_banner", "-encoders"], stderr=subprocess.STDOUT).decode("utf-8", errors="replace")
    except FileNotFoundError as e:
        raise RuntimeError("ffmpeg not found. Install ffmpeg with AV1 support.") from e
    enc = "libsvtav1" if "libsvtav1" in out else ("libaom-av1" if "libaom-av1" in out else None)
    if enc is None:
        raise RuntimeError("No AV1 encoder found in ffmpeg (need libsvtav1 or libaom-av1).")
    _AV1_ENCODER = enc
    return enc


def _open_ffmpeg_rawvideo_writer(path: Path, width: int, height: int, fps: int) -> "subprocess.Popen[bytes]":
    """Open an ffmpeg process that accepts raw RGB24 frames on stdin."""
    import subprocess

    path.parent.mkdir(parents=True, exist_ok=True)
    codec = _pick_av1_encoder()
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "pipe:0",
        "-c:v",
        codec,
        "-crf",
        "30",
        "-pix_fmt",
        "yuv420p",
        "-f",
        "mp4",
        str(path),
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def _close_ffmpeg_writer(proc: "subprocess.Popen[bytes]") -> None:
    """Close stdin, wait, raise on ffmpeg failure."""
    if proc.stdin is not None:
        try:
            proc.stdin.close()
        except Exception:
            pass
    stderr = b""
    if proc.stderr is not None:
        try:
            stderr = proc.stderr.read()
            proc.stderr.close()
        except Exception:
            stderr = b""
    rc = proc.wait()
    if rc != 0:
        msg = stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"ffmpeg exit {rc}: {msg[:500]}")


class RunningStats:
    def __init__(self, dim: int):
        self.dim = int(dim)
        self.count = 0
        self.sum = np.zeros((self.dim,), dtype=np.float64)
        self.sumsq = np.zeros((self.dim,), dtype=np.float64)
        self.min = np.full((self.dim,), np.inf, dtype=np.float64)
        self.max = np.full((self.dim,), -np.inf, dtype=np.float64)

    def merge_payload(self, payload: Dict[str, Any]) -> None:
        n = int(payload.get("count", 0))
        if n <= 0:
            return
        self.count += n
        self.sum += np.asarray(payload["sum"], dtype=np.float64)
        self.sumsq += np.asarray(payload["sumsq"], dtype=np.float64)
        self.min = np.minimum(self.min, np.asarray(payload["min"], dtype=np.float64))
        self.max = np.maximum(self.max, np.asarray(payload["max"], dtype=np.float64))

    def to_checkpoint_payload(self) -> Dict[str, Any]:
        return {
            "count": self.count,
            "sum": self.sum.tolist(),
            "sumsq": self.sumsq.tolist(),
            "min": self.min.tolist(),
            "max": self.max.tolist(),
        }

    def to_stats_dict(self) -> Dict[str, List[float]]:
        if self.count <= 0:
            return {"min": [], "max": [], "mean": [], "std": []}
        mean = self.sum / float(self.count)
        var = (self.sumsq / float(self.count)) - (mean * mean)
        std = np.sqrt(np.maximum(var, 0.0))
        return {
            "min": self.min.tolist(),
            "max": self.max.tolist(),
            "mean": mean.tolist(),
            "std": std.tolist(),
        }


def _payload_from_rows(x: np.ndarray) -> Dict[str, Any]:
    if x.size == 0:
        return {"count": 0}
    x64 = np.asarray(x, dtype=np.float64)
    return {
        "count": int(x64.shape[0]),
        "sum": np.sum(x64, axis=0).tolist(),
        "sumsq": np.sum(x64 * x64, axis=0).tolist(),
        "min": np.min(x64, axis=0).tolist(),
        "max": np.max(x64, axis=0).tolist(),
    }


def _configure_tf_env() -> None:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")


def load_droid_dataset(source_dir: Path, max_episodes: Optional[int]):
    _configure_tf_env()
    import tensorflow_datasets as tfds  # type: ignore

    for name in ("droid", "droid_100"):
        try:
            print(f"Trying to load TFDS dataset '{name}' from data_dir={source_dir}...", flush=True)
            ds = tfds.load(
                name,
                data_dir=str(source_dir),
                split="train",
                read_config=tfds.ReadConfig(try_autocache=False),
            )
            print(f"Loaded dataset '{name}'.")
            if max_episodes is not None:
                ds_to_use = ds.take(max_episodes)
            else:
                ds_to_use = ds
            return tfds.as_numpy(ds_to_use), name
        except Exception as e:  # noqa: BLE001
            print(f"  Failed to load '{name}': {e}")
    raise RuntimeError("Could not load DROID RLDS dataset via tensorflow_datasets.")


def extract_language_for_episode(step_lang: np.ndarray) -> str:
    for s in step_lang:
        try:
            txt = s.decode("utf-8") if isinstance(s, (bytes, bytearray)) else str(s)
        except Exception:  # noqa: BLE001
            continue
        if txt:
            return txt
    return ""


def derive_episode_id(episode: Dict[str, Any], episode_index: int) -> str:
    meta = episode.get("episode_metadata", {})
    for key in ("episode_id", "trajectory_path", "file_path", "recording_folderpath"):
        val = meta.get(key)
        if val is None:
            continue
        if isinstance(val, (bytes, bytearray)):
            return val.decode("utf-8")
        try:
            import numpy as _np  # type: ignore
            if isinstance(val, _np.ndarray):
                elem = val.item() if val.size == 1 else val.flat[0]
                if isinstance(elem, (bytes, bytearray)):
                    return elem.decode("utf-8")
                return str(elem)
        except Exception:  # noqa: BLE001
            pass
        return str(val)
    return f"episode_{episode_index:06d}"


def our_path_to_relative(stored_id: str) -> Optional[str]:
    """
    Convert stored episode id (e.g. full path to trajectory.h5 or canonical id) to the
    relative path form used as value in episode_id_to_path (e.g. site/outcome/date/folder).
    """
    s = stored_id.strip().rstrip("/")
    if s.endswith("/trajectory.h5"):
        s = s[: -len("/trajectory.h5")]
    elif s.endswith("trajectory.h5"):
        s = s[: -len("trajectory.h5")].rstrip("/")
    parts = [p for p in s.split("/") if p]
    if len(parts) >= 4:
        return "/".join(parts[-4:])
    if len(parts) >= 1:
        return "/".join(parts)
    return None


def resolve_canonical_episode_id(
    stored_id: str,
    path_to_episode_id: Dict[str, str],
    episode_id_to_path: Dict[str, str],
) -> Optional[str]:
    """Resolve RLDS-derived stored_id to canonical episode_id used in annotation files."""
    rel = our_path_to_relative(stored_id)
    if rel is not None:
        canonical = path_to_episode_id.get(rel)
        if canonical is not None:
            return canonical
    if stored_id in episode_id_to_path:
        return stored_id
    return None


def build_both_cameras_episode_set(superset_path: Path) -> set:
    """
    Load cam2base_extrinsic_superset.json and return set of episode_ids that have
    extrinsics for both left and right cameras (>= 2 camera-serial keys with 6-float list).
    """
    with superset_path.open("r") as f:
        superset = json.load(f)
    allowed = set()
    for episode_id, entry in superset.items():
        if not isinstance(entry, dict):
            continue
        count = 0
        for k, v in entry.items():
            if k in ("relative_path",) or k.endswith("_metric_type") or k.endswith("_quality_metric") or k.endswith("_source"):
                continue
            if k.isdigit() and isinstance(v, list) and len(v) == 6 and all(isinstance(x, (int, float)) for x in v):
                count += 1
        if count >= 2:
            allowed.add(episode_id)
    return allowed


def episode_to_table_and_videos(
    episode: Dict[str, Any],
    episode_index: int,
) -> Tuple[pa.Table, Dict[str, np.ndarray], Dict[str, Any]]:
    """
    Returns (parquet_table_no_images, empty video dict, episode_meta).

    Note: video writing is handled elsewhere to avoid materializing (T,H,W,3) arrays in RAM.
    This function is kept for backward compatibility but no longer produces video arrays.
    """
    raise RuntimeError("episode_to_table_and_videos is deprecated; use episode_to_table_and_write_videos.")


def episode_to_table_and_write_videos(
    episode: Dict[str, Any],
    episode_index: int,
    output_path: Path,
    chunk_str: str,
) -> Tuple[pa.Table, Dict[str, Any]]:
    """
    Build parquet table and stream-write MP4s without stacking all frames in memory.
    """
    steps_iter = episode["steps"]
    episode_id = derive_episode_id(episode, episode_index=episode_index)

    # Collect tabular fields
    frame_index: List[int] = []
    state_rows: List[np.ndarray] = []
    action_rows: List[np.ndarray] = []
    reward_vals: List[float] = []
    done_vals: List[bool] = []

    has_state = True
    has_action = True
    has_reward = True
    has_done = True

    language_instruction = ""

    # Video writers opened lazily on first frame to infer shape
    writers: Dict[str, Any] = {}

    def _video_path(cam_name: str) -> Path:
        return output_path / "videos" / chunk_str / cam_name / f"episode_{episode_index:06d}.mp4"

    cam_map = {
        "observation.images.wrist_image_left": ("observation", "wrist_image_left"),
        "observation.images.exterior_image_1_left": ("observation", "exterior_image_1_left"),
        "observation.images.exterior_image_2_left": ("observation", "exterior_image_2_left"),
    }

    num_steps = 0
    for i, s in enumerate(steps_iter):
        num_steps += 1
        frame_index.append(i)

        if not language_instruction:
            v = s.get("language_instruction", b"")
            try:
                txt = v.decode("utf-8") if isinstance(v, (bytes, bytearray)) else str(v)
            except Exception:
                txt = ""
            if txt:
                language_instruction = txt

        obs = s.get("observation", {})

        # Stream-write videos (one frame at a time)
        for cam_name, (obs_root, obs_key) in cam_map.items():
            if obs_root != "observation":
                continue
            frame = obs.get(obs_key)
            if frame is None:
                continue
            frame_u8 = np.asarray(frame, dtype=np.uint8)
            if frame_u8.ndim != 3 or frame_u8.shape[-1] != 3:
                continue
            if cam_name not in writers:
                h, w = int(frame_u8.shape[0]), int(frame_u8.shape[1])
                writers[cam_name] = _open_ffmpeg_rawvideo_writer(_video_path(cam_name), width=w, height=h, fps=FPS)
            proc = writers[cam_name]
            if proc.stdin is None:
                raise RuntimeError("ffmpeg stdin is not available.")
            try:
                proc.stdin.write(frame_u8.tobytes())
            except BrokenPipeError as e:
                raise RuntimeError(f"ffmpeg pipe broke while writing {cam_name}") from e

        # Tabular fields (match previous behavior: if any step missing, drop the whole column)
        jp = obs.get("joint_position", None)
        if has_state:
            if jp is None:
                has_state = False
            else:
                state_rows.append(np.asarray(jp, dtype=np.float32))

        act = s.get("action", None)
        if has_action:
            if act is None:
                has_action = False
            else:
                action_rows.append(np.asarray(act, dtype=np.float32))

        r = s.get("reward", None)
        if has_reward:
            if r is None:
                has_reward = False
            else:
                reward_vals.append(float(r))

        d = s.get("is_last", None)
        if has_done:
            if d is None:
                has_done = False
            else:
                done_vals.append(bool(d))

    if num_steps == 0:
        raise ValueError("Episode has zero steps.")

    # Finalize videos
    for proc in writers.values():
        _close_ffmpeg_writer(proc)

    if not language_instruction:
        language_instruction = ""

    cols: Dict[str, Any] = {
        "episode_index": np.full(num_steps, episode_index, dtype=np.int64),
        "frame_index": np.asarray(frame_index, dtype=np.int64),
        "episode_id": np.array([episode_id] * num_steps, dtype=object),
        "language_instruction": np.array([language_instruction] * num_steps, dtype=object),
        "timestamp": (np.arange(num_steps, dtype=np.float32) / float(FPS)),
        "index": np.arange(num_steps, dtype=np.int64),
    }

    if has_state and len(state_rows) == num_steps:
        cols["observation.state"] = list(state_rows)
    if has_action and len(action_rows) == num_steps:
        cols["action"] = list(action_rows)
    if has_reward and len(reward_vals) == num_steps:
        cols["next.reward"] = np.asarray(reward_vals, dtype=np.float32)
    if has_done and len(done_vals) == num_steps:
        cols["next.done"] = np.asarray(done_vals, dtype=bool)

    table = pa.Table.from_pandas(pd.DataFrame(cols), preserve_index=False)
    episode_meta = {
        "episode_index": episode_index,
        "length": int(num_steps),
        "episode_id": episode_id,
        "language_instruction": language_instruction,
        "has_state": has_state and len(state_rows) == num_steps,
        "has_action": has_action and len(action_rows) == num_steps,
    }
    return table, episode_meta


def write_mp4(path: Path, frames: np.ndarray, fps: int = FPS) -> None:
    """
    Write (T, H, W, 3) uint8 RGB to MP4 using AV1 codec, yuv420p, 15fps.
    Matches official lerobot/droid_100: AV1 has no macroblock constraint, so 320x180 is encoded as-is.
    Uses ffmpeg with libaom-av1 (or libsvtav1 if available) and CRF 30.
    """
    import subprocess

    path.parent.mkdir(parents=True, exist_ok=True)
    T, H, W, C = frames.shape
    # Prefer libsvtav1 (faster); fall back to libaom-av1
    for codec in ("libsvtav1", "libaom-av1"):
        try:
            cmd = [
                "ffmpeg", "-y",
                "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps),
                "-i", "pipe:0",
                "-c:v", codec, "-crf", "30", "-pix_fmt", "yuv420p",
                "-f", "mp4", str(path),
            ]
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            # Write raw RGB frames to stdin (row-major, contiguous)
            proc.stdin.write(frames.astype(np.uint8).tobytes())
            proc.stdin.close()
            stderr = proc.stderr.read().decode("utf-8", errors="replace")
            proc.stderr.close()
            if proc.wait() != 0:
                raise RuntimeError(f"ffmpeg exit {proc.returncode}: {stderr[:500]}")
            return
        except FileNotFoundError:
            continue
        except Exception as e:  # noqa: BLE001
            if codec == "libaom-av1":
                raise RuntimeError(f"Failed to write MP4 with AV1: {e}") from e
            continue
    raise RuntimeError("ffmpeg not found or no AV1 encoder (libsvtav1/libaom-av1) available. Install ffmpeg with AV1 support.")


def _process_one_episode(
    episode: Dict[str, Any],
    out_idx: int,
    task_index: int,
    output_dir: str,
    canonical_id: Optional[str],
    stored_id: str,
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Worker: convert one episode to table + videos, write parquet and MP4s, return meta and stats payloads.
    Module-level so it can be pickled for ProcessPoolExecutor.
    """
    output_path = Path(output_dir)
    chunk_id = out_idx // EPISODES_PER_CHUNK
    chunk_str = f"chunk-{chunk_id:03d}"
    table, epi_meta = episode_to_table_and_write_videos(episode, episode_index=out_idx, output_path=output_path, chunk_str=chunk_str)
    epi_meta["episode_index"] = out_idx
    epi_meta["task_index"] = task_index
    epi_meta["episode_id"] = stored_id
    if canonical_id is not None:
        epi_meta["canonical_episode_id"] = canonical_id
    task_col = np.full(table.num_rows, task_index, dtype=np.int64)
    table = table.append_column("task_index", pa.array(task_col))
    data_chunk_dir = output_path / "data" / chunk_str
    data_chunk_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, data_chunk_dir / f"episode_{out_idx:06d}.parquet")
    state_payload: Optional[Dict[str, Any]] = None
    action_payload: Optional[Dict[str, Any]] = None
    if "observation.state" in table.column_names and table.num_rows > 0:
        state_arr = np.stack([np.asarray(table.column("observation.state")[i], dtype=np.float32) for i in range(table.num_rows)])
        state_payload = _payload_from_rows(state_arr)
    if "action" in table.column_names and table.num_rows > 0:
        action_arr = np.stack([np.asarray(table.column("action")[i], dtype=np.float32) for i in range(table.num_rows)])
        action_payload = _payload_from_rows(action_arr)
    return (epi_meta, state_payload, action_payload)


def write_meta_v2(
    output_dir: Path,
    episodes_meta: List[Dict[str, Any]],
    task_to_instruction: Dict[int, str],
    stats: Dict[str, Dict[str, List[float]]],
    total_frames: int,
    conversion_meta: Optional[Dict[str, Any]] = None,
) -> None:
    meta_dir = output_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    total_episodes = len(episodes_meta)
    total_tasks = len(task_to_instruction)
    num_chunks = (total_episodes + EPISODES_PER_CHUNK - 1) // EPISODES_PER_CHUNK

    info = {
        "codebase_version": "v2.0",
        "robot_type": "droid",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": total_tasks,
        "total_chunks": num_chunks,
        "chunks_size": EPISODES_PER_CHUNK,
        "fps": FPS,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_id": {"dtype": "str", "shape": [], "names": None},
            "language_instruction": {"dtype": "str", "shape": [], "names": None},
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
            # observation.state: joint motors motor_0..motor_6, shape [7]
            "observation.state": {"dtype": "float32", "shape": [7], "names": None},
            "action": {"dtype": "float32", "shape": [7], "names": None},
            "next.reward": {"dtype": "float32", "shape": [1], "names": None},
            "next.done": {"dtype": "bool", "shape": [1], "names": None},
        },
    }
    for cam in CAMERA_KEYS:
        info["features"][cam] = {
            "dtype": "video",
            "shape": list(IMAGE_SHAPE),
            "names": ["height", "width", "channel"],
            "video_info": {
                "video.fps": FPS,
                "video.codec": "av1",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False,
            },
        }
    with (meta_dir / "info.json").open("w") as f:
        json.dump(info, f, indent=2)

    with (meta_dir / "tasks.jsonl").open("w") as f:
        for task_index in range(total_tasks):
            instr = task_to_instruction[task_index]
            f.write(
                json.dumps(
                    {
                        "task_index": task_index,
                        "task": instr,
                        "language_instruction": instr,
                    }
                )
                + "\n"
            )

    with (meta_dir / "episodes.jsonl").open("w") as f:
        for m in episodes_meta:
            # Match droid_100: store task texts under \"tasks\" (list), plus length
            f.write(
                json.dumps(
                    {
                        "episode_index": m["episode_index"],
                        "tasks": [m["language_instruction"]],
                        "length": m["length"],
                    }
                )
                + "\n"
            )

    with (meta_dir / "stats.json").open("w") as f:
        json.dump(stats, f, indent=2)

    # Output index -> original identity (canonical_id when available for annotation lookup)
    index_to_id: Dict[str, Any] = {}
    for m in episodes_meta:
        idx = str(m["episode_index"])
        if "canonical_episode_id" in m:
            index_to_id[idx] = {
                "canonical_id": m["canonical_episode_id"],
                "stored_id": m["episode_id"],
            }
        else:
            index_to_id[idx] = m["episode_id"]
    with (meta_dir / "episode_index_to_id.json").open("w") as f:
        json.dump(index_to_id, f, indent=2)

    if conversion_meta is not None:
        with (meta_dir / "conversion_meta.json").open("w") as f:
            json.dump(conversion_meta, f, indent=2)


def scan_completed_out_idxs(output_dir: Path) -> set:
    """Return set of out_idxs that already have a written parquet file."""
    done: set = set()
    data_dir = output_dir / "data"
    if not data_dir.exists():
        return done
    for p in data_dir.glob("chunk-*/episode_*.parquet"):
        try:
            idx = int(p.stem.split("_")[1])
            done.add(idx)
        except (IndexError, ValueError):
            pass
    return done


def load_checkpoint(output_dir: Path) -> Optional[Dict[str, Any]]:
    path = output_dir / "meta" / "progress_checkpoint.json"
    if not path.exists():
        return None
    with path.open("r") as f:
        return json.load(f)


def save_checkpoint(
    output_dir: Path,
    episodes_meta: List[Dict[str, Any]],
    task_instruction_to_index: Dict[str, int],
    state_stats: RunningStats,
    action_stats: RunningStats,
    total_frames: int,
    included_canonical_ids: Optional[List[str]] = None,
    conversion_meta: Optional[Dict[str, Any]] = None,
) -> None:
    """Atomically write progress_checkpoint.json and refresh all meta/ files."""
    meta_dir = output_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    checkpoint: Dict[str, Any] = {
        "episodes_meta": episodes_meta,
        "task_instruction_to_index": task_instruction_to_index,
        "state_stats": state_stats.to_checkpoint_payload(),
        "action_stats": action_stats.to_checkpoint_payload(),
        "total_frames": total_frames,
    }
    if included_canonical_ids is not None:
        checkpoint["included_canonical_ids"] = included_canonical_ids

    tmp_path = meta_dir / "progress_checkpoint.json.tmp"
    final_path = meta_dir / "progress_checkpoint.json"
    with tmp_path.open("w") as f:
        json.dump(checkpoint, f)
    tmp_path.rename(final_path)

    # Also refresh the full meta/ directory with current partial state
    task_to_instruction = {idx: lang for lang, idx in task_instruction_to_index.items()}
    stats: Dict[str, Dict[str, List[float]]] = {}
    if state_stats.count > 0:
        stats["observation.state"] = state_stats.to_stats_dict()
    if action_stats.count > 0:
        stats["action"] = action_stats.to_stats_dict()
    write_meta_v2(output_dir, episodes_meta, task_to_instruction, stats, total_frames, conversion_meta=conversion_meta)
    print(f"[checkpoint] Saved: {len(episodes_meta)} episodes, {total_frames} frames.", flush=True)


def main() -> None:
    print("Starting DROID conversion ...", flush=True)
    parser = argparse.ArgumentParser(description="Convert DROID RLDS to LeRobot v2 format (chunked data + videos + meta).")
    parser.add_argument("--source-dir", type=str, default="/ocean/projects/cis240058p/hbhatia1/data/droid_raw_small")
    parser.add_argument("--output-dir", type=str, default="/ocean/projects/cis240058p/hbhatia1/data/droid_v2_small")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument(
        "--cam2base-superset",
        type=str,
        default=None,
        help="Path to cam2base_extrinsic_superset.json. If set, only convert episodes with both left and right camera extrinsics.",
    )
    parser.add_argument(
        "--annotations-dir",
        type=str,
        default=None,
        help="Directory containing episode_id_to_path.json. Defaults to parent of --cam2base-superset when that is set.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of parallel workers for conversion (default 1). Use 30-50 for many CPUs.",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="If set, enable Weights & Biases logging to this project name.",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="Optional run name for wandb (default: auto from output_dir and timestamp).",
    )
    args = parser.parse_args()

    source_dir = Path(os.path.expanduser(args.source_dir))
    output_dir = Path(os.path.expanduser(args.output_dir))

    use_wandb = args.wandb_project is not None and wandb is not None
    if args.wandb_project is not None and wandb is None:
        print("Warning: --wandb-project set but wandb not installed. Install with: pip install wandb")
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or f"convert_{output_dir.name}_{int(time.time())}",
            config={
                "source_dir": str(source_dir),
                "output_dir": str(output_dir),
                "max_episodes": args.max_episodes,
                "num_workers": args.num_workers,
                "cam2base_superset": args.cam2base_superset,
                "annotations_dir": args.annotations_dir,
            },
            job_type="dataset_conversion",
        )

    t_start = time.perf_counter()

    allowed_episode_ids: Optional[set] = None
    path_to_episode_id: Optional[Dict[str, str]] = None
    episode_id_to_path: Optional[Dict[str, str]] = None
    superset_path: Optional[Path] = None

    if args.cam2base_superset is not None:
        superset_path = Path(os.path.expanduser(args.cam2base_superset))
        if not superset_path.exists():
            raise FileNotFoundError(f"cam2base-superset not found: {superset_path}")
        annotations_dir = Path(os.path.expanduser(args.annotations_dir)) if args.annotations_dir else superset_path.parent
        if not annotations_dir.exists():
            raise FileNotFoundError(f"annotations-dir not found: {annotations_dir}")
        episode_id_to_path_path = annotations_dir / "episode_id_to_path.json"
        if not episode_id_to_path_path.exists():
            raise FileNotFoundError(f"episode_id_to_path.json not found in {annotations_dir}")
        with episode_id_to_path_path.open("r") as f:
            episode_id_to_path = json.load(f)
        path_to_episode_id = {v: k for k, v in episode_id_to_path.items()}
        allowed_episode_ids = build_both_cameras_episode_set(superset_path)
        print(f"Filter: only episodes with both cameras (from {superset_path.name}). Allowed set size: {len(allowed_episode_ids)}")

    num_workers = max(1, int(args.num_workers))
    episodes, ds_name = load_droid_dataset(source_dir, max_episodes=args.max_episodes)
    print(f"Converting dataset '{ds_name}' -> {output_dir} (LeRobot v2 layout) ... (workers={num_workers})")

    episodes_meta: List[Dict[str, Any]] = []
    task_instruction_to_index: Dict[str, int] = {}
    state_stats = RunningStats(dim=7)
    action_stats = RunningStats(dim=7)
    total_frames = 0
    included_canonical_ids: List[str] = []

    # --- Resume support: load checkpoint and scan already-written parquets ---
    done_out_idxs: set = scan_completed_out_idxs(output_dir)
    checkpoint = load_checkpoint(output_dir)
    if checkpoint is not None and done_out_idxs:
        print(f"Resuming from checkpoint: {len(done_out_idxs)} episodes already done.", flush=True)
        episodes_meta = checkpoint["episodes_meta"]
        task_instruction_to_index = checkpoint["task_instruction_to_index"]
        total_frames = checkpoint["total_frames"]
        state_stats.merge_payload(checkpoint["state_stats"])
        action_stats.merge_payload(checkpoint["action_stats"])
        if "included_canonical_ids" in checkpoint:
            included_canonical_ids = checkpoint["included_canonical_ids"]
    elif done_out_idxs:
        print(f"Found {len(done_out_idxs)} completed parquets but no checkpoint; skipping them (stats restart from 0).", flush=True)
    # -------------------------------------------------------------------------

    if num_workers <= 1:
        # Single-threaded: one pass over TFDS
        # out_idx_counter tracks the global output index independently of episodes_meta length
        # so that skipped (already-done) episodes advance the counter correctly.
        out_idx_counter = len(episodes_meta)
        for epi_idx, episode in enumerate(episodes):
            stored_id = derive_episode_id(episode, epi_idx)
            if allowed_episode_ids is not None and path_to_episode_id is not None and episode_id_to_path is not None:
                canonical = resolve_canonical_episode_id(stored_id, path_to_episode_id, episode_id_to_path)
                if canonical is None or canonical not in allowed_episode_ids:
                    continue
            else:
                canonical = None

            out_idx = out_idx_counter
            out_idx_counter += 1

            if out_idx in done_out_idxs:
                print(f"Skipping episode {epi_idx} -> output index {out_idx} (already done).", flush=True)
                if canonical is not None and canonical not in included_canonical_ids:
                    included_canonical_ids.append(canonical)
                continue

            if canonical is not None:
                included_canonical_ids.append(canonical)

            print(f"Processing episode {epi_idx} -> output index {out_idx} ...")
            chunk_id = out_idx // EPISODES_PER_CHUNK
            chunk_str = f"chunk-{chunk_id:03d}"
            table, epi_meta = episode_to_table_and_write_videos(episode, episode_index=out_idx, output_path=output_dir, chunk_str=chunk_str)
            epi_meta["episode_index"] = out_idx
            if canonical is not None:
                epi_meta["canonical_episode_id"] = canonical
            lang = epi_meta["language_instruction"]
            if lang not in task_instruction_to_index:
                task_instruction_to_index[lang] = len(task_instruction_to_index)
            epi_meta["task_index"] = task_instruction_to_index[lang]
            episodes_meta.append(epi_meta)
            total_frames += epi_meta["length"]

            task_col = np.full(table.num_rows, epi_meta["task_index"], dtype=np.int64)
            table = table.append_column("task_index", pa.array(task_col))

            data_chunk_dir = output_dir / "data" / chunk_str
            data_chunk_dir.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, data_chunk_dir / f"episode_{out_idx:06d}.parquet")

            if "observation.state" in table.column_names and table.num_rows > 0:
                state_arr = np.stack([np.asarray(table.column("observation.state")[i], dtype=np.float32) for i in range(table.num_rows)])
                state_stats.merge_payload(_payload_from_rows(state_arr))
            if "action" in table.column_names and table.num_rows > 0:
                action_arr = np.stack([np.asarray(table.column("action")[i], dtype=np.float32) for i in range(table.num_rows)])
                action_stats.merge_payload(_payload_from_rows(action_arr))

            if (out_idx + 1) % CHECKPOINT_INTERVAL == 0:
                _conv_meta_partial: Optional[Dict[str, Any]] = None
                if superset_path is not None and included_canonical_ids:
                    _conv_meta_partial = {
                        "filter": "cam2base_superset_both_cameras",
                        "cam2base_superset_path": str(superset_path.resolve()),
                        "included_canonical_episode_ids": included_canonical_ids,
                    }
                save_checkpoint(output_dir, episodes_meta, task_instruction_to_index, state_stats, action_stats, total_frames, included_canonical_ids=included_canonical_ids, conversion_meta=_conv_meta_partial)

            if use_wandb and (out_idx + 1) % 50 == 0:
                wandb.log(
                    {"progress/episodes": out_idx + 1, "progress/total_frames": total_frames},
                    step=out_idx + 1,
                )
    else:
        # Multi-worker: pass 1 = build task index and work list; pass 2 = process with pool
        def _language_from_episode(ep: Dict[str, Any]) -> str:
            # Pass 1 only: we re-load the dataset for pass 2, so it's OK to consume ep["steps"] here.
            for s in ep["steps"]:
                v = s.get("language_instruction", b"")
                try:
                    txt = v.decode("utf-8") if isinstance(v, (bytes, bytearray)) else str(v)
                except Exception:
                    txt = ""
                if txt:
                    return txt
            return ""

        # Pass 1: build episode_idx -> (out_idx, task_index, canonical_id, stored_id)
        # task_instruction_to_index may already be pre-populated from checkpoint; keep it to stay consistent.
        episode_idx_to_work: Dict[int, Tuple[int, int, Optional[str], str]] = {}
        print("Pass 1: building task index and work list ...")
        for epi_idx, episode in enumerate(episodes):
            stored_id = derive_episode_id(episode, epi_idx)
            if allowed_episode_ids is not None and path_to_episode_id is not None and episode_id_to_path is not None:
                canonical = resolve_canonical_episode_id(stored_id, path_to_episode_id, episode_id_to_path)
                if canonical is None or canonical not in allowed_episode_ids:
                    continue
                included_canonical_ids.append(canonical)
                canonical_id: Optional[str] = canonical
            else:
                canonical_id = None
            out_idx = len(episode_idx_to_work)
            lang = _language_from_episode(episode)
            if lang not in task_instruction_to_index:
                task_instruction_to_index[lang] = len(task_instruction_to_index)
            task_index = task_instruction_to_index[lang]
            episode_idx_to_work[epi_idx] = (out_idx, task_index, canonical_id, stored_id)
        total_to_process = len(episode_idx_to_work)
        todo_count = total_to_process - len(done_out_idxs)
        print(f"Pass 1 done: {total_to_process} episodes total, {len(done_out_idxs)} already done, {todo_count} to process with {num_workers} workers.")
        if use_wandb:
            wandb.log({"progress/pass1_episodes": total_to_process})

        # Pass 2: re-load dataset, submit included episodes to pool (skip already-done ones)
        episodes_iter2, _ = load_droid_dataset(source_dir, max_episodes=args.max_episodes)
        results_by_out_idx: Dict[int, Tuple[Dict[str, Any], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]] = {}
        submitted = 0
        completed = 0

        # Build a running state for incremental checkpointing during pass 2.
        # We start from checkpoint data (episodes_meta / stats already loaded above).
        ckpt_episodes_meta: List[Dict[str, Any]] = list(episodes_meta)
        ckpt_state_stats = RunningStats(dim=7)
        ckpt_action_stats = RunningStats(dim=7)
        ckpt_state_stats.merge_payload(state_stats.to_checkpoint_payload())
        ckpt_action_stats.merge_payload(action_stats.to_checkpoint_payload())
        ckpt_total_frames = total_frames

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures: Dict[Any, int] = {}
            max_in_flight = max(1, num_workers * 2)

            def _submit_one(epi_idx: int, episode: Dict[str, Any]) -> None:
                nonlocal submitted
                out_idx, task_index, canonical_id, stored_id = episode_idx_to_work[epi_idx]
                if out_idx in done_out_idxs:
                    return  # Skip already-done episodes
                # Materialize the nested TF dataset (steps) into a plain list so it
                # can be pickled for IPC with ProcessPoolExecutor.
                ep_concrete = dict(episode)
                ep_concrete["steps"] = list(episode["steps"])
                fut = executor.submit(
                    _process_one_episode,
                    ep_concrete,
                    out_idx,
                    task_index,
                    str(output_dir.resolve()),
                    canonical_id,
                    stored_id,
                )
                futures[fut] = out_idx
                submitted += 1
                if submitted % 50 == 0 or submitted == todo_count:
                    print(f"Submitted {submitted}/{todo_count} episodes ...", flush=True)
                if use_wandb and (submitted % 100 == 0 or submitted == todo_count):
                    wandb.log({"progress/submitted": submitted, "progress/total_to_process": todo_count})

            def _collect_one_done() -> None:
                nonlocal completed, ckpt_total_frames
                for fut in as_completed(list(futures.keys()), timeout=None):
                    out_idx = futures.pop(fut)
                    try:
                        epi_meta, state_payload, action_payload = fut.result()
                        results_by_out_idx[out_idx] = (epi_meta, state_payload, action_payload)
                    except Exception as e:
                        raise RuntimeError(f"Worker failed for output index {out_idx}: {e}") from e
                    completed += 1
                    # Update incremental checkpoint state
                    epi_meta_r, state_pl, action_pl = results_by_out_idx[out_idx]
                    ckpt_episodes_meta.append(epi_meta_r)
                    ckpt_total_frames += epi_meta_r["length"]
                    if state_pl is not None:
                        ckpt_state_stats.merge_payload(state_pl)
                    if action_pl is not None:
                        ckpt_action_stats.merge_payload(action_pl)
                    if completed % 50 == 0 or completed == todo_count:
                        print(f"Completed {completed}/{todo_count} episodes ...", flush=True)
                    if use_wandb and (completed % 100 == 0 or completed == todo_count):
                        wandb.log({"progress/completed": completed, "progress/total_to_process": todo_count})
                    if completed % CHECKPOINT_INTERVAL == 0:
                        _conv_meta_partial: Optional[Dict[str, Any]] = None
                        if superset_path is not None and included_canonical_ids:
                            _conv_meta_partial = {
                                "filter": "cam2base_superset_both_cameras",
                                "cam2base_superset_path": str(superset_path.resolve()),
                                "included_canonical_episode_ids": included_canonical_ids,
                            }
                        # Sort before checkpointing to keep episodes.jsonl ordered
                        sorted_ckpt = sorted(ckpt_episodes_meta, key=lambda m: m["episode_index"])
                        save_checkpoint(output_dir, sorted_ckpt, task_instruction_to_index, ckpt_state_stats, ckpt_action_stats, ckpt_total_frames, included_canonical_ids=included_canonical_ids, conversion_meta=_conv_meta_partial)
                    return

            for epi_idx, episode in enumerate(episodes_iter2):
                if epi_idx not in episode_idx_to_work:
                    continue
                _submit_one(epi_idx, episode)
                if len(futures) >= max_in_flight:
                    _collect_one_done()
            while futures:
                _collect_one_done()

        # Merge checkpoint episodes_meta (from previous runs) with newly collected results
        checkpoint_meta_by_idx: Dict[int, Dict[str, Any]] = {m["episode_index"]: m for m in episodes_meta}
        for out_idx in sorted(results_by_out_idx.keys()):
            epi_meta, state_payload, action_payload = results_by_out_idx[out_idx]
            checkpoint_meta_by_idx[out_idx] = epi_meta
            total_frames += epi_meta["length"]
            if state_payload is not None:
                state_stats.merge_payload(state_payload)
            if action_payload is not None:
                action_stats.merge_payload(action_payload)
        episodes_meta = [checkpoint_meta_by_idx[i] for i in sorted(checkpoint_meta_by_idx.keys())]

    if allowed_episode_ids is not None and len(episodes_meta) == 0:
        print("Warning: filter enabled but no episodes passed the filter. Output dataset will be empty.")

    stats: Dict[str, Dict[str, List[float]]] = {}
    if state_stats.count > 0:
        stats["observation.state"] = state_stats.to_stats_dict()
    if action_stats.count > 0:
        stats["action"] = action_stats.to_stats_dict()

    task_to_instruction = {idx: lang for lang, idx in task_instruction_to_index.items()}
    conversion_meta: Optional[Dict[str, Any]] = None
    if superset_path is not None and included_canonical_ids:
        conversion_meta = {
            "filter": "cam2base_superset_both_cameras",
            "cam2base_superset_path": str(superset_path.resolve()),
            "included_canonical_episode_ids": included_canonical_ids,
        }
    write_meta_v2(output_dir, episodes_meta, task_to_instruction, stats, total_frames, conversion_meta=conversion_meta)
    duration_sec = time.perf_counter() - t_start
    print(f"Done. Wrote {len(episodes_meta)} episodes, {total_frames} frames to {output_dir}.")

    if use_wandb:
        summary = {
            "total_episodes": len(episodes_meta),
            "total_frames": total_frames,
            "total_tasks": len(task_to_instruction),
            "duration_sec": duration_sec,
            "frames_per_sec": total_frames / duration_sec if duration_sec > 0 else 0,
        }
        for key, d in stats.items():
            for stat_name, vals in d.items():
                summary[f"stats/{key}/{stat_name}"] = vals
        wandb.log(summary)
        wandb.finish()


if __name__ == "__main__":
    main()
