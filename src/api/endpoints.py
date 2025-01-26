from fastapi import APIRouter, UploadFile, File, HTTPException
from schemas import PredictionResponse
from functools import lru_cache
from typing import Dict, Any
import tensorflow as tf
from PIL import Image
import numpy as np
import io
import yaml

router = APIRouter()


@lru_cache
def load_config() -> Dict[str, Any]:
    """
    Load and cache the configuration file.

    Returns:
        Dict[str, Any]: The configuration dictionary loaded from the YAML file.
    """
    PATH = "/media/ahmed/Files/Brain-Project/configs/config.yaml"
    with open(PATH, "r") as file:
        return yaml.safe_load(file)


@lru_cache
def get_model() -> tf.keras.Model:
    """
    Load and cache the TensorFlow model from the path specified in the config.

    Returns:
        tf.keras.Model: The loaded TensorFlow model.
    """

    config = load_config()
    return tf.keras.models.load_model(config["models"]["checkpoints_path"])


def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Preprocess the image for model prediction.

    Args:
        image (Image.Image): The input image to be preprocessed.

    Returns:
        np.ndarray: The preprocessed image as a numpy array.
    """

    image = image.resize((224, 224))

    image_array = np.array(image) / 255.0

    return np.expand_dims(image_array, axis=0)


@router.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    """
    Predict the class of the uploaded image using the pre-trained model.

    Args:
        file (UploadFile): The image file uploaded by the user.

    Returns:
        PredictionResponse: The response containing the filename and the prediction.

    Raises:
        HTTPException: If an error occurs during prediction.
    """

    try:
        image_content = await file.read()

        image = Image.open(io.BytesIO(image_content))

        if image.mode != "RGB":
            image = image.convert("RGB")

        processed_image = preprocess_image(image)

        model = get_model()

        prediction = model.predict(processed_image, batch_size=1)[0][0]

        return PredictionResponse(
            filename=file.filename, prediction=round(float(prediction))
        )

    except Exception as e:
        return HTTPException(status_code=500, detail=str(e))
