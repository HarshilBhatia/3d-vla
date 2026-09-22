from .rlbench import RLBenchDepth2Cloud


def fetch_depth2cloud(dataset_name, scene_frame="world"):
    dataset_name = dataset_name.lower()
    if 'peractcollected' in dataset_name:
        return RLBenchDepth2Cloud((128, 128), scene_frame)
    if 'peract2' in dataset_name:
        return RLBenchDepth2Cloud((256, 256), scene_frame)
    if 'rlbench' in dataset_name:
        return RLBenchDepth2Cloud((256, 256), scene_frame)
    if 'orbital' in dataset_name:
        return RLBenchDepth2Cloud((256, 256), scene_frame)
    return None
