
---

# Breast Cancer Classification with EfficientNetB3

## Project Overview

This project uses a deep learning approach to classify breast ultrasound images into three categories: benign, malignant, and normal. The model is built using TensorFlow and Keras, leveraging the EfficientNetB3 architecture for image classification. The dataset used for training and testing is the Breast Ultrasound Images Dataset.

## Directory Structure

```plaintext
├── BreastCancerModel.h5      # Saved model file
├── README.md                 # This file
└── script.py                 # Python script for training and evaluating the model
```

## Dataset

The dataset is organized into three categories: `benign`, `malignant`, and `normal`. Each category contains grayscale images representing different conditions of breast tissues.

- **Dataset Path**: `/kaggle/input/breast-ultrasound-images-dataset/Dataset_BUSI_with_GT`
- **Categories**: `benign`, `malignant`, `normal`

## Dependencies

Make sure you have the following libraries installed:

```bash
pip install tensorflow opencv-python pillow matplotlib seaborn scikit-learn
```

## How to Run the Code

1. **Load and Prepare the Data:**

   The images are loaded from the dataset directory and displayed using a yellowish colormap for visualization.

   ```python
   data_dir = '/kaggle/input/breast-ultrasound-images-dataset/Dataset_BUSI_with_GT'
   ```

2. **Generate Data Paths with Labels:**

   File paths and labels are generated and combined into a DataFrame for easier manipulation.

3. **Train-Test Split:**

   The data is split into training and testing sets with a ratio of 80:20, maintaining the stratification based on labels.

4. **Create Data Generators:**

   ImageDataGenerator is used to create training and testing generators.

5. **Model Architecture:**

   The model uses the EfficientNetB3 architecture with additional layers for fine-tuning. The model is compiled with the Adamax optimizer and trained for 50 epochs.

   ```python
   base_model = tf.keras.applications.efficientnet.EfficientNetB3(include_top= False, weights= "imagenet", input_shape= img_shape, pooling= 'max')
   ```

6. **Training the Model:**

   The model is trained using the training generator, with accuracy and loss evaluated on the testing set.

7. **Evaluation:**

   The model's performance is evaluated using the confusion matrix and classification report.

   - **Train Accuracy:** 99.92%
   - **Test Accuracy:** 94.30%

8. **Saving the Model:**

   The trained model is saved as `BreastCancerModel.h5`.

   ```python
   model.save('BreastCancerModel.h5')
   ```

## Visualization

- **Confusion Matrix:** Displays the confusion matrix with a custom colormap (`coolwarm`).
- **Classification Report:** Shows precision, recall, f1-score, and support for each class.

## Results

The model achieved a test accuracy of 94.30%, with high precision and recall for all classes.
