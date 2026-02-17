import logging 
from pathlib import Path 
from datetime import datetime


LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

def get_logger(name: str):

    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file_path = LOG_DIR / f"{run_id}.log"   

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    file_hanlder = logging.FileHandler(log_file_path)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    file_hanlder.setFormatter(formatter)

    logger.addHandler(file_hanlder)

    return logger