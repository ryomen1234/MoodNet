from torch.utils.data import DataLoader
from src.features.dataset import MoodNetDataset
from pathlib import Path
from torchvision import transforms


def get_dataloader(
        root_dir: Path,
        batch_size: int = 32,
        transform = None       
):
    train_dataset = MoodNetDataset(
        root_dir / "train",
        transform
   )
    
    test_dataset = MoodNetDataset(
        root_dir / "test",
        transform
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