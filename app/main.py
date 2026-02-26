from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from src.inference.data_validation import ImageValidator
from src.inference.inference import ImageClassifier

app = FastAPI()

model_path = Path(r"E:\project_archive\MoodNet\artifacts\models\ResNet18.pth")

if not model_path.exists():
    raise FileNotFoundError(f"File not found: {model_path}")



# Load model once at startup
classifier = ImageClassifier(model_path=model_path, num_classes=7)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    try:
        # Validate type
        ImageValidator.validate_file_type(file.content_type)

        file_bytes = await file.read()

        # Validate image integrity
        ImageValidator.validate_image_bytes(file_bytes)

        # Predict
        result = classifier._predict(file_bytes)

        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception:
        raise HTTPException(status_code=500, detail="Internal Server Error")
