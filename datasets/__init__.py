from .rlbench import (
    Peract2Dataset,
    PeractDataset,
    PeractCollectedDataset,
    HiveformerDataset,
    OrbitalWristDataset,
)


def fetch_dataset_class(dataset_name):
    """Fetch the dataset class based on the dataset name."""
    dataset_classes = {
        "Peract2_3dfront_3dwrist": Peract2Dataset,

        "Peract": PeractDataset,
        
        "PeractCollected": PeractCollectedDataset,
        "HiveformerRLBench": HiveformerDataset,
        "OrbitalWrist": OrbitalWristDataset,
    }
    
    if dataset_name not in dataset_classes:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return dataset_classes[dataset_name]
