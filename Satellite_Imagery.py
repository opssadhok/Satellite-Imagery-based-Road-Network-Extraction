#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


import random
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

def prepare_data(data_dir, target_size=(512, 512)):
    
    images = []
    masks = []

    files = sorted(os.listdir(data_dir))
    image_files = [file for file in files if 'sat' in file and file.endswith('.jpg')]
    mask_files = {file.replace('_mask.png', ''): file for file in files if 'mask' in file and file.endswith('.png')}

    for img_file in image_files:
        img_path = os.path.join(data_dir, img_file)
        image = cv2.imread(img_path)
        image = cv2.resize(image, target_size)
        image = image.astype('float32') / 255.0  # Normalize to [0, 1]
        images.append(image)

        base_name = img_file.replace('_sat.jpg', '')
        if base_name in mask_files:
            mask_path = os.path.join(data_dir, mask_files[base_name])
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, target_size)
            mask = (mask > 128).astype(np.uint8)  # Binarize the mask
            masks.append(mask)

    return np.array(images), np.array(masks)

def image_label_generator(dataset_path, batch_size=32, target_size=(512, 512)):
   
    files = sorted(os.listdir(dataset_path))
    image_files = [file for file in files if 'sat' in file and file.endswith('.jpg')]
    label_files = [file for file in files if 'mask' in file and file.endswith('.png')]

    while True:
        batch_images = []
        batch_labels = []
        
        for _ in range(batch_size):
            idx = random.randint(0, len(image_files) - 1)
            
            img_path = os.path.join(dataset_path, image_files[idx])
            image = cv2.imread(img_path)
            image = cv2.resize(image, target_size)
            image = image.astype('float32') / 255.0  # Normalize to [0, 1]
            batch_images.append(image)

            label_path = os.path.join(dataset_path, label_files[idx])
            label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            label = cv2.resize(label, target_size)
            label = (label > 128).astype(np.uint8)  # Binarize the label
            batch_labels.append(label)
        
        yield np.array(batch_images), np.array(batch_labels)

# Ensure random seed for reproducibility
random.seed(42)

# Specify the path to your dataset directory
dataset_path = 'D:\\D- ops_sadhok\\train'

# Load images and masks
images, masks = prepare_data(dataset_path)

# Split the data into training and validation sets
train_images, val_images, train_masks, val_masks = train_test_split(images, masks, test_size=0.2, random_state=42)

# Ensure masks are of type float32 for model compatibility
train_masks = train_masks.astype('float32')
val_masks = val_masks.astype('float32')

# Create a generator
generator = image_label_generator(dataset_path, batch_size=1)

# Generate a batch of images and labels
image_batch, label_batch = next(generator)

# Extract the image and mask from the batch
selected_image = image_batch[0]
selected_label = label_batch[0]

plt.figure(figsize=(14, 8))

# Display the satellite image
plt.subplot(121)
plt.imshow(selected_image)
plt.title("Satellite Image")

# Display the mask image
plt.subplot(122)
plt.imshow(selected_label, cmap='gray')  

# Use grayscale colormap for mask
plt.title("Mask Image")

plt.show()


# In[ ]:


from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall

def unet_model(input_size=(512, 512, 3)):
    inputs = Input(input_size)

    # U-Net architecture with adjustments for input and output size
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same')(conv5)

    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(up6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(up7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(up8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(up9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy', Precision(), Recall()])

    return model

# Define and compile the model
model = unet_model()

# Fit the model with original-size masks
model.fit(
    train_images, 
    train_masks, 
    validation_data=(val_images, val_masks), 
    epochs=1, 
    batch_size=4
)


# In[ ]:


from tensorflow.keras.callbacks import ModelCheckpoint

# Define the checkpoint callback
checkpoint_callback = ModelCheckpoint(
    r'D:\D- ops_sadhok\road_network_extraction_model.h5',  # Path to save the model
    save_best_only=True,  # Save only the best model based on validation loss
    monitor='val_loss',  # Monitor validation loss
    mode='min',  # Mode to save the best model
    verbose=1  # Verbosity mode
)

# Train the model with the checkpoint callback
history = model.fit(
    train_images, 
    train_masks, 
    validation_data=(val_images, val_masks), 
    epochs=20, 
    batch_size=4,
    callbacks=[checkpoint_callback]  # Include the checkpoint callback
)


# In[ ]:


# Save the model
model.save(r'D:\D- ops_sadhok\road_network_extraction_model.h5')

# Confirm that the model is saved
print("Model saved to 'D:\\D- ops_sadhok\\road_network_extraction_model.h5'")


# In[ ]:


import os

# Verify the saved model file
if os.path.exists(r'D:\D- ops_sadhok\road_network_extraction_model.h5'):
    print("Model file exists.")
else:
    print("Model file does not exist.")


# In[ ]:


import os
import numpy as np
from keras.models import load_model
import cv2
from PIL import Image
from patchify import patchify
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Function to preprocess a single image
def preprocess_image(image, patch_size=256):
    size_x = (image.shape[1] // patch_size) * patch_size
    size_y = (image.shape[0] // patch_size) * patch_size
    image = Image.fromarray(image)
    image = image.crop((0, 0, size_x, size_y))
    image = np.array(image)
    return image

# Function to process image patches
def process_patches(image, patch_size=256):
    patched_images = patchify(image, (patch_size, patch_size, 3), step=patch_size)
    minmaxscaler = MinMaxScaler()
    processed_patches = []

    for i in range(patched_images.shape[0]):
        for j in range(patched_images.shape[1]):
            patch = patched_images[i, j, :, :]
            patch = minmaxscaler.fit_transform(patch.reshape(-1, patch.shape[-1])).reshape(patch.shape)
            processed_patches.append(patch.astype(np.float32))

    return np.array(processed_patches)

# Load the pre-trained model
model = load_model('/path/to/your/model/road_network_extraction_model.h5')

# Load and preprocess test data
test_dataset_root_folder = 'D:\\D- ops_sadhok\\test'  # Update with your test dataset folder
test_image_patch_size = 256

# List test image files
test_image_files = [f for f in os.listdir(test_dataset_root_folder) if f.endswith('_sat.jpg')]

# Initialize lists to store images and predictions
original_images = []
predicted_masks = []

# Process up to 20 images
for test_image_file in test_image_files[:20]:
    test_image_path = os.path.join(test_dataset_root_folder, test_image_file)

    # Load the image
    test_image = cv2.imread(test_image_path, 1)
    if test_image is not None:
        # Preprocess the image
        preprocessed_image = preprocess_image(test_image, patch_size=test_image_patch_size)
        # Process the patches
        test_image_patches = process_patches(preprocessed_image, patch_size=test_image_patch_size)

        # Use the model to make predictions
        predictions = model.predict(test_image_patches)

        # Convert predictions to binary masks
        binary_predictions = (predictions > 0.5).astype(np.uint8)

        # Store one of the original images and its prediction
        original_images.append(test_image)
        # Use the first prediction for simplicity
        predicted_masks.append(binary_predictions[0])

# Visualize results for the 20 images
plt.figure(figsize=(20, 10))
for i in range(len(original_images)):
    plt.subplot(4, 10, 2*i + 1)
    plt.imshow(cv2.cvtColor(original_images[i], cv2.COLOR_BGR2RGB))
    plt.title(f'Original {i+1}')
    plt.axis('off')

    plt.subplot(4, 10, 2*i + 2)
    plt.imshow(predicted_masks[i], cmap='gray')
    plt.title(f'Predicted {i+1}')
    plt.axis('off')

plt.tight_layout()
plt.show()


# In[ ]:




