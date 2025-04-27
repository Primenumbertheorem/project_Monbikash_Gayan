# Waste Classification Model  using CNN

This repository contains a deep learning model for classifying images into two categories. The model is built using PyTorch and is designed to work with custom datasets. Below, you'll find details about the model architecture, input/output specifications, and how to use the model for training and inference.

---


## Model Architecture

The model is implemented in [`model.py`](model.py) as the `MyCustomModel` class. It consists of the following layers:

1. **Feature Extraction**:
   - 3 Convolutional layers with Batch Normalization, ReLU activation, and MaxPooling.
   - Input size: `(3, 128, 128)` (3 channels, 128x128 resolution).
   - Output size after feature extraction: `(128, 16, 16)`.

2. **Classification**:
   - Flatten layer to convert feature maps into a vector.
   - Fully connected layer with 512 neurons and ReLU activation.
   - Dropout layer with a rate of 0.3 to prevent overfitting.
   - Final fully connected layer with 2 output neurons (for binary classification).

---

## Input and Output

### Input
- **Image Input**: The model expects RGB images with dimensions `(128, 128)` (resized automatically during preprocessing).
- **Batch Input**: A batch of images with shape `(batch_size, 3, 128, 128)`.

### Output
- **Predicted Class**: The model outputs a single integer representing the predicted class:
  - `0`: Class 0 (e.g., "Category O").
  - `1`: Class 1 (e.g., "Category R").

---

## Training the Model

To train the model, use the `my_descriptively_named_train_function` function in [`train.py`](train.py). The training process includes:
- Loading the dataset from the paths specified in [`config.py`](config.py).
- Using the Adam optimizer and CrossEntropyLoss.
- Saving the trained model weights to the specified path.


