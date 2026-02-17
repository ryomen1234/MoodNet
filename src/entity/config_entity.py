from pathlib import Path
from dataclasses import dataclass
from typing import List

@dataclass
class DataIngestionConfig:
    source_data_path: Path
    raw_data_path: Path

@dataclass
class DataValidationConfig:
    splits: List[str]
    allowed_extensions: List[str]
    classes: List[str]
