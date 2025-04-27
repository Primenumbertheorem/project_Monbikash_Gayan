# Waste Classification Model Using CNN


This repository contains a deep learning model for classifying waste images into two categories-Organic(O) and Recyclable(R). The model is built using PyTorch and is designed to work with custom datasets. Below, there are details about the model architecture, input/output specifications, and how to use the model for training and inference.

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

## Dataset

The dataset used for training and testing the model can be downloaded from Kaggle:  
[Waste Classification Dataset](https://www.kaggle.com/datasets/techsash/waste-classification-data/data)

Ensure the dataset is organized in the format required by `torchvision.datasets.ImageFolder`:
```
dataset/
├── TRAIN/
│   ├── Class0/
│   └── Class1/
└── TEST/
    ├── Class0/
    └── Class1/
```

---

## Training the Model

To train the model, use the `my_descriptively_named_train_function` function in [`train.py`](train.py). The training process includes:
- Loading the dataset from the paths specified in [`config.py`](config.py).
- Using the Adam optimizer and CrossEntropyLoss.
- Saving the trained model weights to the specified path.

### Example Command
```bash
python train.py
```

---

## Inference

To make predictions on new images, use the `cryptic_inf_f` function in [`predict.py`](predict.py). This function:
1. Loads the trained model weights.
2. Preprocesses the input image to match the model's requirements.
3. Outputs the predicted class.

### Example Usage
```python
from predict import cryptic_inf_f

image_path = "path/to/image.jpg"
predicted_class = cryptic_inf_f(image_path)
print(f"Predicted Class: {predicted_class}")
```

---

## Dependencies

The project requires the following Python libraries:
- `torch`
- `torchvision`
- `Pillow`

Install the dependencies using:
```bash
pip install torch torchvision Pillow
```

---



## Conclusion

This project demonstrates the use of Convolutional Neural Networks (CNNs) for waste classification. The model is lightweight and can be trained on custom datasets. It can be extended to classify more categories or integrated into real-world applications like waste sorting systems, promoting sustainable waste management practices.

