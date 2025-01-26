from typing import List, Any, Tuple
import albumentations as A
import numpy as np


class ImageAugmentation:
    """
    A class for augmenting images and their corresponding labels using Albumentations.

    Attributes:
        images (List[np.ndarray]): A list of input images as numpy arrays.
        labels (List[Any]): A list of labels corresponding to the input images.
        transform (A.Compose): An Albumentations composition of transformations to apply.
    """

    def __init__(self, images: List[np.ndarray], labels: List[Any]) -> None:
        """
        Initializes the ImageAugmentation class with images and labels.

        Args:
            images (List[np.ndarray]): A list of input images as numpy arrays.
            labels (List[Any]): A list of labels corresponding to the input images.
        """
        self.images = images
        self.labels = labels

        self.transform = A.Compose(
            [
                A.RandomRotate90(p=0.3),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.ShiftScaleRotate(p=0.2),
                A.RandomBrightnessContrast(p=0.2),
            ]
        )

    def augment(self) -> Tuple[List[np.ndarray], List[Any]]:
        """
        Augments the images and their corresponding labels.

        Returns:
            Tuple[List[np.ndarray], List[Any]]: A tuple containing:
                - A list of augmented images as numpy arrays.
                - A list of corresponding labels.
        """
        augmented_images = []
        augmented_labels = []

        for image, label in zip(self.images, self.labels):
            augmented_images.append(image)
            augmented_labels.append(label)

            for _ in range(3):
                augmented = self.transform(image=image)["image"]
                augmented_images.append(augmented)
                augmented_labels.append(label)

        return augmented_images, augmented_labels
