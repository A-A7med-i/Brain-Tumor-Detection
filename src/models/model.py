import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from typing import Tuple, Dict, Any, Optional, List
from tensorflow.keras.applications import VGG16, ResNet50, EfficientNetB0
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    Activation,
    BatchNormalization,
    Input,
)


class BinaryImageClassifier:
    """
    A class for building, training, and evaluating a binary image classification model.

    Attributes:
        input_shape (Tuple[int, int, int]): The shape of the input images (height, width, channels).
        learning_rate (float): The learning rate for the optimizer.
        base_model (tf.keras.Model): The base model used for transfer learning.
        model (tf.keras.Model): The compiled binary classification model.
    """

    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (224, 224, 3),
        base_model_name: str = "ResNet50",
        learning_rate: float = 0.001,
    ) -> None:
        """
        Initializes the BinaryImageClassifier class.

        Args:
            input_shape (Tuple[int, int, int]): The shape of the input images. Default is (224, 224, 3).
            base_model_name (str): The name of the base model to use. Options: "VGG16", "ResNet50", "EfficientNetB0".
                                  Default is "ResNet50".
            learning_rate (float): The learning rate for the optimizer. Default is 0.001.
        """
        self.input_shape = input_shape
        self.learning_rate = learning_rate
        self.base_model = self._get_base_model(base_model_name)
        self.model = self._initialize_model()

    def _get_base_model(self, name: str) -> tf.keras.Model:
        """
        Retrieves the base model for transfer learning.

        Args:
            name (str): The name of the base model.

        Returns:
            tf.keras.Model: The base model with the top classification layer excluded.
        """
        base_models = {
            "VGG16": VGG16,
            "ResNet50": ResNet50,
            "EfficientNetB0": EfficientNetB0,
        }

        return base_models[name](include_top=False, input_shape=self.input_shape)

    def _initialize_model(self) -> tf.keras.Model:
        """
        Initializes the binary classification model by adding custom layers on top of the base model.

        Returns:
            tf.keras.Model: The compiled binary classification model.
        """
        self.base_model.trainable = False

        inputs = Input(shape=self.input_shape)

        x = self.base_model(inputs)
        x = GlobalAveragePooling2D()(x)
        x = BatchNormalization()(x)

        x = Dense(128)(x)
        x = Activation("relu")(x)
        x = Dropout(0.3)(x)

        x = Dense(64)(x)
        x = Activation("relu")(x)
        x = Dropout(0.3)(x)

        outputs = Dense(1, activation="sigmoid")(x)

        return tf.keras.Model(inputs, outputs)

    def setup_model(
        self, optimizer: Optional[tf.keras.optimizers.Optimizer] = None
    ) -> None:
        """
        Compiles the model with the specified optimizer and metrics.

        Args:
            optimizer (Optional[tf.keras.optimizers.Optimizer]): The optimizer to use.
                                                                Default is Adam with the specified learning rate.
        """
        self.model.compile(
            optimizer=optimizer or Adam(learning_rate=self.learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy", "Precision", "Recall"],
        )

    def get_callbacks(
        self, checkpoint_path: str, patience: int = 5, monitor: str = "val_loss"
    ) -> List[tf.keras.callbacks.Callback]:
        """
        Returns a list of callbacks for training the model.

        Args:
            checkpoint_path (str): The path to save the best model checkpoint.
            patience (int): The number of epochs to wait before reducing the learning rate or stopping training.
                           Default is 5.
            monitor (str): The metric to monitor for early stopping and learning rate reduction.
                          Default is "val_loss".

        Returns:
            List[tf.keras.callbacks.Callback]: A list of callbacks.
        """
        return [
            ReduceLROnPlateau(
                monitor=monitor, factor=0.2, patience=patience, min_lr=1e-6
            ),
            EarlyStopping(
                monitor=monitor, patience=patience, restore_best_weights=True
            ),
            ModelCheckpoint(
                filepath=checkpoint_path,
                monitor=monitor,
                mode="min",
                save_best_only=True,
            ),
        ]

    def fit_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        checkpoint_path: str,
        epochs: int = 15,
        batch_size: int = 32,
        validation_split: float = 0.2,
    ) -> tf.keras.callbacks.History:
        """
        Trains the model on the provided training data.

        Args:
            X_train (np.ndarray): The training data.
            y_train (np.ndarray): The training labels.
            checkpoint_path (str): The path to save the best model checkpoint.
            epochs (int): The number of epochs to train. Default is 15.
            batch_size (int): The batch size for training. Default is 32.
            validation_split (float): The fraction of the training data to use for validation. Default is 0.2.

        Returns:
            tf.keras.callbacks.History: The training history.
        """
        history = self.model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=self.get_callbacks(checkpoint_path),
        )
        return history

    def evaluate_model(
        self, X_test: np.ndarray, y_test: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluates the model on the provided test data.

        Args:
            X_test (np.ndarray): The test data.
            y_test (np.ndarray): The test labels.

        Returns:
            Dict[str, float]: A dictionary containing the evaluation metrics.
        """
        results = self.model.evaluate(X_test, y_test)

        metrics = {
            "loss": round(results[0], 3),
            "accuracy": round(results[1], 3),
            "precision": round(results[2], 3),
            "recall": round(results[3], 3),
        }

        return metrics

    def plot_training_history(
        self, history: tf.keras.callbacks.History, figsize=(14, 6)
    ) -> None:
        """
        Plots the training and validation accuracy and loss.

        Args:
            history (tf.keras.callbacks.History): The training history.
            figsize (Tuple[int, int]): The size of the plot. Default is (14, 6).
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        ax1.plot(
            history.history["accuracy"], label="Train", color="#1f77b4", linewidth=2
        )

        ax1.plot(
            history.history["val_accuracy"],
            label="Validation",
            color="#ff7f0e",
            linestyle="--",
        )

        ax1.set_title("Model Accuracy")
        ax1.set_ylabel("Accuracy")
        ax1.set_xlabel("Epoch")

        ax1.grid(True, alpha=0.3)
        ax1.legend()

        ax2.plot(history.history["loss"], label="Train", color="#2ca02c", linewidth=2)

        ax2.plot(
            history.history["val_loss"],
            label="Validation",
            color="#d62728",
            linestyle="--",
        )

        ax2.set_title("Loss")
        ax2.set_ylabel("Loss")
        ax2.set_xlabel("Epoch")

        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()
        plt.show()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generates predictions for the input data.

        Args:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The predicted probabilities.
        """
        return self.model.predict(X)

    def save_model(self, filepath: str) -> None:
        """
        Saves the model to the specified filepath.

        Args:
            filepath (str): The path to save the model.
        """
        self.model.save(filepath)

    def load_model(self, filepath: str) -> None:
        """
        Loads a model from the specified filepath.

        Args:
            filepath (str): The path to load the model from.
        """
        self.model = tf.keras.models.load_model(filepath)

    def get_model_summary(self) -> None:
        """
        Prints a summary of the model architecture.
        """
        return self.model.summary()
