from torch.utils.data import DataLoader
from src.features.dataset import MoodNetDataset
from pathlib import Path


def get_dataloader(
        root_dir: Path,
        batch_size: int = 32
):
    train_dataset = MoodNetDataset(
        root_dir / "train",
        transform=None
   )
    
    test_dataset = MoodNetDataset(
        root_dir= root_dir / "test",
        transform=None 
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, test_loader