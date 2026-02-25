from src.features.dataloader import get_dataloader
from pathlib import Path
import yaml

from src.features import transform

def dataloader_pipeline():

    config_file = Path("config/config.yaml")

    if not config_file.exists():
        raise FileNotFoundError("Config file not found")

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    raw_dir = Path(config["data_ingestion"]["raw_data_dir"])
    batch_size = 32
    
    trf = transform.get_transform(train=True)
    train_loader, test_loader = get_dataloader(raw_dir, batch_size, trf)

    # images, labels = next(iter(train_dataloader))

    # print(f"Image shape: {images.shape}")
    # print(f"Labels shape: {labels.shape}")
    # print(f"First label: {labels[0].item()}")

    return train_loader, test_loader


if __name__ == "__main__":
    dataloader_pipeline()