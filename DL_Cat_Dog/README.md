## Cat vs. Dog Classification Using CNN and Transfer Learning

- **Project Overview:**
  This project focuses on building a binary classification model to distinguish between images of cats and dogs.
- **It utilizes two approaches:**
  1. **Custom Convolutional Neural Network (CNN)**
  2. **Transfer Learning using MobileNetV2 and VGG16**
- **Both models are trained, validated, and tested on the Cat and Dog Images Dataset to achieve robust performance in classifying images.**


- **Dataset**
    
  - **Dataset Name:** Cat and Dog Images for Classification
  - **Source:** Kaggle Dataset
  - **Format:** A .csv file with image paths and labels (0 for cat, 1 for dog).
- **Preprocessing:**
  - **Images are resized to 64X64 pixels.**
  - **Data augmentation techniques such as rotation, width/height shift, shear, zoom, and horizontal flipping are applied to the training set.**

# Custom CNN

- **Architecture:**
  - Three Convolutional layers with ReLU activation.
  - Batch Normalization and MaxPooling for regularization and dimensionality reduction.
  - Fully connected layers with a final output layer using the sigmoid activation function.
  - Dropout applied to prevent overfitting.

- **Compilation:**
  
  - **Optimizer:** Adam with a learning rate of 0.0001.
  - **Loss Function:** categorical Cross-Entropy.
  - **Metrics:** Accuracy.
    
# Transfer Learning (MobileNetV2)
  
- **Base Model: Pretrained MobileNetV2 on ImageNet (weights frozen).**
  - **Additional layers include:**
        Flattening, dense layers with regularization (l2), and dropout.
        Final sigmoid activation for binary classification.
  - **Advantages:** Leverages a pretrained model for faster convergence and higher accuracy.

- **Model Evaluation**
  **Metrics:**
    - Training and validation accuracy.
    - Training and validation loss.

### Results

| Model         | Test Accuracy | Test Loss |
|---------------|---------------|-----------|
| Custom CNN    | ~88%          | ~0.38     |
| MobileNetV2   | ~49%          | ~1.41     |
| VGG16         | ~50%          | ~0.69     |

- **Custom CNN:** A lightweight model with competitive accuracy.

## Usage: Training and Prediction

1. **Train the Models**:  
   Follow the provided training code to train `model1` (Custom CNN) and `model2` (MobileNetV2). Ensure the dataset is properly preprocessed before initiating training.  

2. **Display Results:**
The predict_and_display function will process the input image, predict its class using the selected model, and display the image along with the predicted class.

### Conclusion
This project demonstrates the effectiveness of CNNs and transfer learning in binary image classification tasks.


## 📓 Colab Notebook  
  - Explore the implementation in this project using the (https://www.kaggle.com/datasets/ashfakyeafi/cat-dog-images-for-classification/data)
