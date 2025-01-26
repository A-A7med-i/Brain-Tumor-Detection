# Brain Tumor Image Classification Project

This project focuses on building a binary image classification model to classify brain tumor images using deep learning techniques. The project includes data preprocessing, model training, evaluation, and visualization utilities.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Installation](#installation)
3. [Running the API](#running-the-api)
4. [Project Structure](#project-structure)
5. [Contributing](#contributing)
6. [License](#license)

---

## Project Overview

The goal of this project is to classify brain tumor images into two categories: **Tumor** (Yes) or **No Tumor** (No). The project uses transfer learning with pre-trained models like ResNet50, VGG16, and EfficientNetB0, and includes utilities for data preprocessing, model training, evaluation, and visualization.

---

## Installation

To set up the project, follow these steps:

1. **Clone the repository**:
   ```bash
    git clone https://github.com/A-A7med-i/Brain-Tumor-Detection.git
    cd brain-tumor-classification
    ```

2. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Download the dataset**:
    * Place your dataset in the data/raw directory.


---

## Running the API

To run the FastAPI API for real-time predictions, follow these steps:

1. **Navigate to the `src` directory**:
    ```bash
    cd src
    ```

2. **Start the FastAPI server**:
    * Run main.py
    

3. **Access the API**:
    * Open your browser or use a tool like curl or Postman.
    * Send a POST request to http://0.0.0.0:8000/predict with an image file to get predictions.


* Example using curl:
    ```bash
    curl -X POST -F "file=@path/to/your/image.jpg" http://0.0.0.0:8000/predict
    ```

* Example response:
    ```json
    {
        "filename": "image.jpg",
        "prediction": 1
    }
    ```

---


## Project Structure

The project is organized as follows:

```python
deep_learning_project/
│
├── data/                      # Data directory
│   ├── raw/                   # Original data
│   └── processed/             # Processed data
│
├── models/                    # Model storage
│   └── checkpoints/           # Model checkpoints
│
├── src/                      # Source code
│   ├── __init__.py
│   ├── data/                # Data operations
│   │   ├── __init__.py
│   │   ├── augmentation.py  # Data augmentation
│   │   ├── make_dataset.py  # Dataset creation
│   │   └── preprocess.py    # Data preprocessing
│   │
│   ├── models/              # Model definitions
│   │   ├── __init__.py
│   │   └── model.py         # Model architectures
│   │
│   ├── utils/               # Utility functions
│   │   ├── __init__.py
│   │   └── helper.py        # Helper functions
│   │
│   ├── visualization/       # Visualization tools
│   │   ├── __init__.py
│   │   └── plot.py          # Plotting functions
│   │     
│   └── api/                
│       ├── __init__.py
│       ├── main.py         # FastAPI app
│       ├── endpoints.py    # API routes
│       └── schemas.py      # Pydantic models
│
├── configs/                # Configuration files
│   └── config.yaml         # Project configuration
│
├── notebooks/               # Jupyter notebooks
│   ├── EDA.ipynb            # Exploratory Data Analysis
│   ├── exploratory.ipynb    # Data exploration
│   └── experiments.ipynb    # Experiments
│
├── requirements.txt     # Project dependencies
├── setup.py             # Package setup
├── LICENSE              # LICENSE File
├── .gitignore           # Git ignore rules
└── README.md            # Project documentation
```


---

## Contributing

Contributions are welcome! If you'd like to contribute, please follow these steps:

1. **Fork the repository**:
   ```bash
    git clone https://github.com/A-A7med-i/Brain-Tumor-Detection.git
    cd brain-tumor-classification
    ```
2. **Create a new branch for your feature or bugfix**:
    ```bash
    git checkout -b feature/your-feature-name
    ```

3. **Commit your changes**:
    ```bash
    git add .
    git commit -m "Add your commit message here"
    ```

4. **Push your changes to your forked repository**:
    ```bash
    git push origin feature/your-feature-name
    ```

5. **Submit a pull request to the main repository.**


---

## License

This project is licensed under the [MIT License](LICENSE). See the LICENSE file for details.