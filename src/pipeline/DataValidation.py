import yaml
from pathlib import Path
from src.data.data_validation import DataValidation
from src.entity.config_entity import DataIngestionConfig, DataValidationConfig

def validation_pipeline():
    config_file = Path("config/config.yaml")

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    
    ingest_config = DataIngestionConfig(
        source_data_path=Path(config["data_ingestion"]["source_data_path"]),
        raw_data_path=Path(config["data_ingestion"]["raw_data_dir"])
    )

    valid_config = DataValidationConfig(
        splits=config["data_validation"]["splits"],
        allowed_extensions=config["data_validation"]["allowed_extension"],
        classes=config["data_validation"]["classes"]

    )

    data_validate = DataValidation(valid_config, ingest_config)
    return data_validate.init_validation()

if __name__ == "__main__":
    report = validation_pipeline()
    print(report)
    
    