import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image

from src.utils.logger import get_logger


logger = get_logger(__name__)

class MoodNetDataset(Dataset):

    def __init__(self, root_dir: Path, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}

        classes = sorted([d.name for d in root_dir.iterdir() if root_dir.is_dir()])
        self.class_to_idx = {cls: i for i, cls in enumerate(classes)}
        
        for cls in classes:
            class_path = self.root_dir / cls 
            for img in class_path.iterdir():
                self.samples.append((img, self.class_to_idx[cls]))

    def __len__(self) -> int:
        return len(self.samples) 

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label)