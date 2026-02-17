from src.entity.config_entity import DataValidationConfig, DataIngestionConfig
from src.utils.logger import get_logger
from typing import List, Dict
from tqdm import tqdm
import logging

logger = get_logger(__name__)

if not logger:
    raise Exception("Logger not initiated")

class DataValidation:

    def __init__(
        self,
        validation_config: DataValidationConfig,
        ingestion_config: DataIngestionConfig
    ) -> None:

        self.raw_dir = ingestion_config.raw_data_path
        self.splits = validation_config.splits
        self.allowed_extensions = validation_config.allowed_extensions
        self.classes = validation_config.classes

    def validate_raw_dir(self):
        if not self.raw_dir.exists():
            logger.error(f"Raw directory does not exist: {self.raw_dir}")
            raise FileNotFoundError(f"Missing raw directory: {self.raw_dir}")

    def validate_splits(self):
        logger.info("Validating data splits")

        for split in self.splits:
            split_path = self.raw_dir / split
            if not split_path.exists():
                logger.error(f"Missing split folder: {split_path}")
                raise FileNotFoundError(f"Missing split: {split}")

        logger.info("Split validation complete")

    def get_classes(self, split: str) -> List[str]:
        split_path = self.raw_dir / split
        return sorted(
            d.name for d in split_path.iterdir() if d.is_dir()
        )

    def validate_classes(self, split: str):
        logger.info(f"Validating classes in {split}")

        detected_classes = self.get_classes(split)

        if set(detected_classes) != set(self.classes):
            logger.error(
                f"Class inconsistency in {split}. "
                f"Expected: {self.classes}, Found: {detected_classes}"
            )
            raise ValueError(f"Class mismatch in {split}")

        logger.info("Class validation complete")

    def validate_img(self, split: str, cls: str) -> int:
        class_path = self.raw_dir / split / cls

        imgs = [f for f in class_path.iterdir() if f.is_file()]

        if not imgs:
            logger.error(f"Empty class folder: {class_path}")
            raise ValueError(f"Empty folder: {class_path}")

        for img in tqdm(
            imgs,
            desc=f"Validating {split}/{cls}",
            unit="img",
            leave=False
        ):
            if img.suffix.lower() not in self.allowed_extensions:
                logger.error(f"Invalid image format: {img}")
                raise ValueError(f"Invalid image format: {img}")
        
        return len(imgs)

    def init_validation(self):

        logger.info("Starting data validation")
        logger.info(f"Raw data path: {self.raw_dir}")
        logger.info(f"Splits: {self.splits}")
        logger.info(f"Allowed extensions: {self.allowed_extensions}")
        logger.info(f"Classes: {self.classes}")
        
        report: Dict[str, Dict[str, int]] = {}

        self.validate_raw_dir()
        self.validate_splits()

        for split in self.splits:
            self.validate_classes(split)
            report[split] = {}
            for cls in self.classes:
                count = self.validate_img(split, cls)
                report[split][cls] = count

        logger.info("Data validation completed successfully ✅")
        
        return report
