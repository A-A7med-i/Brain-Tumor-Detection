from sklearn.model_selection import train_test_split
from ..utils.helper import load_config
from typing import List, Tuple, Any
import numpy as np
import cv2

CONFIG_DATA_PATH = "/media/ahmed/Files/Brain-Project/configs/data_config.yaml"


class ProcessingImages:
    """
    A class for processing images and splitting datasets.

    Attributes:
        images (List[np.ndarray]): A list of input images as numpy arrays.
        processed_images (List[np.ndarray]): A list of processed images after resizing and normalization.
        config (dict): Configuration data loaded from the YAML file.
        RANDOM_STATE (int): Random seed for reproducibility, loaded from the configuration.
    """

    def __init__(self, images: List[np.ndarray]) -> None:
        """
        Initializes the ProcessingImages class with a list of images.

        Args:
            images (List[np.ndarray]): A list of input images as numpy arrays.
        """
        self.images = images
        self.processed_images = None
        self.config = load_config(CONFIG_DATA_PATH)
        self.RANDOM_STATE = self.config["constant"]["random_state"]

    def process(self, size: Tuple[int, int] = (224, 224)) -> List[np.ndarray]:
        """
        Processes the images by resizing and normalizing them.

        Args:
            size (Tuple[int, int]): The target size for resizing the images. Default is (224, 224).

        Returns:
            List[np.ndarray]: A list of processed images as numpy arrays.
        """
        self.processed_images = [
            cv2.resize(image, size).astype(np.float32) / 255.0 for image in self.images
        ]

        return self.processed_images

    def split_data(
        self, X: List[np.ndarray], y: List[Any]
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[Any], List[Any]]:
        """
        Splits the dataset into training and testing sets.

        Args:
            X (List[np.ndarray]): A list of input features (images).
            y (List[Any]): A list of corresponding labels.

        Returns:
            Tuple[List[np.ndarray], List[np.ndarray], List[Any], List[Any]]: A tuple containing:
                - X_train: Training features.
                - X_test: Testing features.
                - y_train: Training labels.
                - y_test: Testing labels.
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.15, random_state=self.RANDOM_STATE, shuffle=True
        )

        return self.X_train, self.X_test, self.y_train, self.y_test
