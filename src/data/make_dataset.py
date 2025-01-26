from concurrent.futures import ThreadPoolExecutor
from typing import Any, Tuple, List, Dict
from pathlib import Path
import cv2


class DatasetMaker:
    """
    A class for loading and processing image datasets from a directory.

    Attributes:
        dir_path (Path): The path to the directory containing the raw image data.
        images (List[Any]): A list of loaded images.
        labels (List[int]): A list of labels corresponding to the loaded images.
    """

    def __init__(self, raw_data_path: str) -> None:
        """
        Initializes the DatasetMaker class with the path to the raw data directory.

        Args:
            raw_data_path (str): The path to the directory containing the raw image data.
        """
        self.dir_path = Path(raw_data_path)
        self.images = []
        self.labels = []

    def _read_image(self, image_path: Path) -> Tuple[Any, int]:
        """
        Reads an image from the given path and assigns a label based on the filename.

        Args:
            image_path (Path): The path to the image file.

        Returns:
            Tuple[Any, int]: A tuple containing the loaded image and its corresponding label.
                            The label is 1 if the filename contains 'Y', otherwise 0.
        """
        image = cv2.imread(str(image_path))
        label = 1 if "Y" in image_path.name else 0
        return image, label

    def load_images(self, num_worker: int = 3) -> Tuple[List[Any], List[int]]:
        """
        Loads images and their labels from the directory using multithreading.

        Args:
            num_worker (int): The number of worker threads to use for loading images. Default is 3.

        Returns:
            Tuple[List[Any], List[int]]: A tuple containing:
                - A list of loaded images.
                - A list of corresponding labels.
        """
        image_paths = [
            path
            for inner_dir in self.dir_path.iterdir()
            for path in inner_dir.glob("*")
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        ]

        with ThreadPoolExecutor(max_workers=num_worker) as executor:
            result = list(executor.map(self._read_image, image_paths))

        self.images, self.labels = zip(*result)

        return list(self.images), list(self.labels)

    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Provides information about the loaded dataset.

        Returns:
            Dict[str, Any]: A dictionary containing:
                - total_images: Total number of images.
                - positive_samples: Number of images with label 1.
                - negative_samples: Number of images with label 0.
                - image_shape: Shape of the first image in the dataset.
        """
        return {
            "total_images": len(self.images),
            "positive_samples": sum(self.labels),
            "negative_samples": len(self.labels) - sum(self.labels),
            "image_shape": self.images[0].shape,
        }
