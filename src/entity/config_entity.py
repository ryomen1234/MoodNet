from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    source_data_path: str
    raw_data_path: str
   
