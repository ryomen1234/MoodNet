import torch 
import torch.nn as nn
from tqdm import tqdm 
from time import time

from src.utils.logger import get_logger

logger = get_logger(__name__)

def train_model(
        model,
        criterion,
        optimizer,
        train_loader,
        validation_loader,
        device,
        epochs: int
):
    
    logger.info("Training and validation started")
    logger.info(f"criterion: {criterion}")
    logger.info(f"optimizer: Adam")
    logger.info(f"device: {device}")
    logger.info(f"number of epochs: {epochs}")

    since = time()

    for epoch in range(epochs):

        model.train()
        train_loss = 0.0
        train_acc = 0.0
        total_train = 0.0
        for inputs, labels in tqdm(train_loader, 
                desc=f"Training | {epoch}"
        ):
           
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            train_loss += loss.item()

            pred = torch.argmax(outputs, dim=1)
            train_acc += (pred == labels).sum().item()
            total_train += labels.size(0)
        
        train_loss = train_loss / len(train_loader)
        train_acc = train_acc / total_train
        
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        with torch.inference_mode():
            for inputs, labels in tqdm(validation_loader,
                     desc=f"Validation | Epoch: {epoch}"
            ):
            
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)

                loss = criterion(outputs, labels)
                val_loss += loss.item()

                pred = torch.argmax(outputs, dim=1)
                val_acc += (pred == labels).sum().item() / len(outputs)
            
            val_loss = val_loss / len(validation_loader)
            val_acc = val_acc / len(validation_loader)
       
        logger.info(
        f"Epoch [{epoch+1}/{epochs}] | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Train Acc: {train_acc:.4f} | "
        f"Val Acc: {val_acc:.4f}"
        )

    return model



            

            



