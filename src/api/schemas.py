from pydantic import BaseModel
from typing import Any


class PredictionResponse(BaseModel):
    """
    Response model for the prediction endpoint.

    Attributes:
        filename (str): The name of the uploaded file.
        prediction (float): The prediction result from the model as a float.
    """

    filename: str
    prediction: Any
