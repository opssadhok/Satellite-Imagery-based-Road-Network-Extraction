# Road Network Extraction from Satellite Images using U-Net
This repository contains the code for a deep learning model based on the U-Net architecture to extract road networks from satellite images.

# Table of Contents:
   - **Installation**
   - **Dataset**
   - **Data Preparation**
   - **Model Architecture**
   - **Training the Model**
   - **Evaluating the Model**
   - **Results**
   - **Acknowledgments**
  

# Installation
Enviroment - Jupyter Notebook
To run the code, you need to have the following libraries installed:
pip install numpy 
pip install opencv-python 
pip install matplotlib 
pip install scikit-learn 
pip install tensorflow keras

# Dataset
The dataset includes satellite images and corresponding binary masks indicating road networks. Images and masks are stored in a single folder without subdirectories.

# Data Preparation
The prepare_data function reads images and masks, resizes them to a target size, and normalizes the images. The masks are binarized.

# Satellite Imagery-Based Road Network Extraction

This project involves developing a machine learning model to extract road networks from satellite images automatically. This capability is critical for creating detailed maps and providing accessibility information, particularly in disaster zones and developing countries, to enhance crisis response efforts.

# Project Overview

The model utilizes a CNN-based U-Net architecture to extract road network from satellite imagery. Given the large size of the dataset and limited computational power, HDF5 files are used to manage data efficiently.

# Dataset

The dataset includes:
- Training Images: 6226 RGB satellite images
- Validation Images: 1243 images
- Test Images: 1101 images (without masks)

Each image is paired with a grayscale mask where:
- White (255) represents road pixels
- Black (0) represents the background

Note: Mask values should be binarized at a threshold of 128. Labels might be less accurate in rural areas, with some small roads intentionally unannotated.

# Data Processing:
   - Ensure reproducibility with a fixed random seed.
   - Randomly select an image index.
   - Extract and preprocess image and mask patches, removing any extra dimensions.

# Data Visualization:
   - Display the satellite image.
   - Display the corresponding mask image.
     
# Load and Split Data:
   - Load data from system.
   - Split the dataset into training, validation, and test sets.

# Model Architecture:
   - The U-Net model is used for segmentation. The architecture consists of a contracting path (encoder) to capture context and a symmetric expanding path (decoder) to enable precise localization.
   -
   - Contracting Path
   - The contracting path follows the typical architecture of a convolutional network. It consists of repeated application of two 3x3 convolutions (unpadded convolutions), each followed by a rectified linear unit (ReLU) and a 2x2 max pooling operation with stride 2 for downsampling. At each downsampling step, the number of feature channels is doubled.
   -
   - Expanding Path
   - Every step in the expansive path consists of an upsampling of the feature map followed by a 2x2 convolution ("up-convolution") that halves the number of feature channels, a concatenation with the correspondingly cropped feature map from the contracting path, and two 3x3 convolutions, each followed by a ReLU. The cropping is necessary due to the loss of border pixels in every convolution..

# Define Callbacks:
  
   from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

   checkpoint = ModelCheckpoint('unet_model.keras', monitor='val_loss', verbose=1, save_best_only=True)
   early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

# Train the Model:
   Train the model using the training and validation datasets:
   The model is trained with a checkpoint callback to save the best model based on validation loss.
 
   history = model.fit(X_train, y_train,
                       validation_data=(X_val, y_val),
                       epochs=50,
                       batch_size=16,
                       callbacks=[checkpoint, early_stopping])
  

# Evaluate Model Performance:
    - Evaluate the model on the validation set during training.
    - Test the model on the split training, validation, and test datasets.

# Model Testing:
    - Load test data.
    - Make predictions on the test set.

# Visualization of Predictions:
    - Randomly select an image index from the test set.
    - Display the original satellite image.
    - Display the predicted mask image.

    ## Road Network Extraction Results

The following images demonstrate the results of our road network extraction model.

!Satellite Image/Mask Image](Screenshot (51).png)

![Performance](Screenshot (52).png)


    
#  Acknowledgments
I want to express my sincere gratitude to everyone who took the time to review and visit this GitHub repository. This project would not have been possible without your valuable feedback and support.

Please note that the code provided here might not be runnable on low computational systems due to its resource-intensive nature. It is highly recommended to execute this project on a high GPU-powered system to achieve better performance and accuracy.

Special thanks to task provider for providing me with the opportunity to develop this model for Road Network Extraction from Satellite Images. Your support and encouragement have been invaluable throughout this journey.

Thank you all once again!
