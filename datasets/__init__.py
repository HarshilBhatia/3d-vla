from .rlbench import (
    Peract2Dataset,
    PeractDataset,
    PeractCollectedDataset,
    HiveformerDataset,
    LiberoDataset,
    OrbitalWristDataset,
    OrbitalPeract2Dataset,
    OrbitalPeract2NoWristDataset,
)


def fetch_dataset_class(dataset_name):
    """Fetch the dataset class based on the dataset name."""
    dataset_classes = {
        "Peract2_3dfront_3dwrist": Peract2Dataset,

        "Peract": PeractDataset,
        
        "PeractCollected": PeractCollectedDataset,
        "Libero": LiberoDataset,
        "HiveformerRLBench": HiveformerDataset,
        "OrbitalWrist": OrbitalWristDataset,
        "OrbitalPeract2": OrbitalPeract2Dataset,
        "OrbitalPeract2NoWrist": OrbitalPeract2NoWristDataset,
    }
    
    if dataset_name not in dataset_classes:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return dataset_classes[dataset_name]
