import torch
import torch.nn  as nn
import torch.optim as optim

from src.models.model import ResNet18
from src.pipeline.DataLoader import dataloader_pipeline
from src.models.train import train_model
from pathlib import Path
import yaml

def _train_pipeline():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, test_loader = dataloader_pipeline()
    config_file = Path("config/config.yaml")

    if not config_file.exists():
        raise FileNotFoundError(f"File does't exist: {config_file}")
    
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    
    epochs = 1
    num_classes = len(config["data_validation"]["classes"])
    lr=0.01
    
    model = ResNet18(num_classes=num_classes).to(device)
 
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    resultant_model = train_model(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        train_loader=train_loader,
        validation_loader=test_loader,
        device=device,
        epochs=epochs
    )

    model_path = Path("artifacts/models")
    model_path.mkdir(parents=True, exist_ok=True)

    model_name = "ResNet18_v2.pth"
    model_save_path = model_path / model_name

    print(f"Saving model to: {model_save_path}")
    torch.save(resultant_model.state_dict(), model_save_path)

if __name__ == "__main__":
    _train_pipeline()








