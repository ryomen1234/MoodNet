import yaml
from pathlib import Path
from src.components.data_ingestion import DataIngestion
from src.entity.config_entity import DataIngestionConfig

def data_pipeline():
    config_file = Path("config/config.yaml")

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    
    ingest_config = DataIngestionConfig(
        source_data_path=config["data_ingestion"]["source_data_path"],
        raw_data_path=config["data_ingestion"]["raw_data_dir"]
    )

    data_ingest = DataIngestion(ingest_config)
    raw_data_dir = data_ingest.initialize_data_ingestion()

    return raw_data_dir

if __name__ == "__main__":
    data_pipeline()