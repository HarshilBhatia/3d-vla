import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_multicam_keypose_example() -> dict:
    """Creates a random input example for the MultiCam Keypose policy."""
    return {
        "observation/state": np.random.rand(8).astype(np.float32),
        "observation/base_image": np.random.randint(256, size=(256, 256, 3), dtype=np.uint8),
        "observation/left_wrist_image": np.random.randint(256, size=(256, 256, 3), dtype=np.uint8),
        "observation/right_wrist_image": np.random.randint(256, size=(256, 256, 3), dtype=np.uint8),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class MultiCamKeyposeInputs(transforms.DataTransformFn):
    """Converts dataset observations to the format expected by pi_0.5.

    Three cameras map to:
      cam 0 -> base_0_rgb   (third-person view)
      cam 1 -> left_wrist_0_rgb
      cam 2 -> right_wrist_0_rgb
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/base_image"])
        left_wrist_image = _parse_image(data["observation/left_wrist_image"])
        right_wrist_image = _parse_image(data["observation/right_wrist_image"])

        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": left_wrist_image,
                "right_wrist_0_rgb": right_wrist_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # Mask third wrist slot for pi0/pi05 (not pi0-FAST).
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class MultiCamKeyposeOutputs(transforms.DataTransformFn):
    """Extracts the 8-dim EEF action from the padded model output."""

    def __call__(self, data: dict) -> dict:
        # Model pads actions to action_dim=32; we recover the first 8 (EEF pose + gripper).
        return {"actions": np.asarray(data["actions"][:, :8])}
