from .clip import load_clip
from .siglip import load_siglip2
from .dino import load_dinov2


def fetch_visual_encoders(model_name):
    if model_name == "clip":
        return load_clip()
    if model_name == "siglip2":
        return load_siglip2()
    if model_name == "dino":
        return load_dinov2()
    return None, None
