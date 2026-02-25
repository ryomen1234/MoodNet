import torch.nn as nn
from torchvision import models
from src.utils.logger import get_logger

logger = get_logger(__name__)

class ResNet18(nn.Module):

    def __init__(self, num_classes: int) -> None:
        super().__init__()

        self.model = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
        )

        logger.info("Loaded pretrained ResNet18")

        # Freeze backbone
        for param in self.model.parameters():
            param.requires_grad = False

        # Replace classifier
        self.model.fc = nn.Linear(
            self.model.fc.in_features,
            num_classes
        )

        logger.info(f"Replaced final layer with {num_classes} classes")

    def forward(self, x):
        return self.model(x)