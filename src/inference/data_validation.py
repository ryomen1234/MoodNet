# src/inference/data_validation.py

from PIL import Image
import io


ALLOWED_TYPES = ["image/jpeg", "image/png", "image/jpg"]

class ImageValidator:

    @staticmethod
    def validate_file_type(content_type: str):
        if content_type not in ALLOWED_TYPES:
            raise ValueError("Invalid file type. Only JPEG and PNG allowed.")

    @staticmethod
    def validate_image_bytes(file_bytes: bytes):
        try:
            Image.open(io.BytesIO(file_bytes)).verify()
        except Exception:
            raise ValueError("Corrupted or invalid image file.")