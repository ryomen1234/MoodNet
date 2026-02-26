import yaml
from PIL import Image
from pathlib import Path
import io

import torch 


from src.models.model import ResNet18
from src.features.transform import get_transform


class ImageClassifier:

    def __init__(self, model_path: Path, num_classes: int) -> None:
        self.model_path = model_path
        self.transform = get_transform(train=False)
        self.num_classes = num_classes
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = self._load_model()   # load once

    def _load_model(self):
        model = ResNet18(num_classes=self.num_classes)
        model.load_state_dict(
            torch.load(self.model_path, map_location=self.device)
        )
        model.to(self.device)
        model.eval()
        return model
    
    def _predict(self, image_bytes: bytes):
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = self.transform(image).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            output = self.model(image)
            pred = torch.argmax(output, dim=1)
        
        return pred.item()