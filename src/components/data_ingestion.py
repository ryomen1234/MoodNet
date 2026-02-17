from tqdm import tqdm 
from src.entity.config_entity import DataIngestionConfig
import zipfile
from pathlib import Path 
from src.utils.logger import get_logger


logger = get_logger(__name__)

if not logger:
    raise Exception(f"Logger not initiated.")

class DataIngestion:
    def __init__(self, config: DataIngestionConfig) -> None:
        self.source_data_dir = Path(config.source_data_path)
        self.raw_data_dir = Path(config.raw_data_path)
    
    def initialize_data_ingestion(self):
        logger.info("Data ingestion started.")
        logger.info(f"Data source path: {self.source_data_dir}")
        logger.info(f"Data extract path: {self.raw_data_dir}")

        if not self.source_data_dir.exists():
            logger.error("Source data dir missing.")
            raise FileNotFoundError("Source data dir missing.")
        
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)

        try:
            with zipfile.ZipFile(self.source_data_dir, "r") as zip_ref:
                files = zip_ref.namelist()
                for file in tqdm(
                    files,
                    desc="Extracting files",
                    unit="file",
                    ncols=100
                ):
                    logger.info(f"Extracting {len(file)} files from zip")
                    zip_ref.extractall(self.raw_data_dir)
        except zipfile.BadZipFile:
            logger.error("Invalid or currupted zip file")
            raise 
        except Exception:
            logger.error("Some unexpected error occured")
            raise

        logger.info("Data ingestion completed successfully")
        return self.raw_data_dir


        
        


