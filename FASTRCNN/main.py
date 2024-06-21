import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    # Only use the first GPU
    tf.config.set_visible_devices(gpus[0], 'GPU')
import os
import numpy as np
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import time
from tensorflow.keras.applications import VGG16

# Update the paths to your directories
local_images_dir = r'C:\Users\malin\Desktop\poze-experiment-disertatie\frames'
local_masks_dir = r'C:\Users\malin\Desktop\poze-experiment-disertatie\masks'

input_image_size = (144, 144)

def read_image(path):
    img = cv2.imread(path)
    img = cv2.resize(img, input_image_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

images_listdir = os.listdir(local_images_dir)

number = 2729
MASKS = np.zeros((number, input_image_size[0], input_image_size[1], 1), dtype=bool)
IMAGES = np.zeros((number, input_image_size[0], input_image_size[1], 3), dtype=np.uint8)

for j, file in enumerate(images_listdir[:number]):
    try:
        image = read_image(os.path.join(local_images_dir, file))
        image_ex = np.expand_dims(image, axis=0)
        IMAGES[j] = image_ex
        mask = read_image(os.path.join(local_masks_dir, file))
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = mask.reshape(input_image_size[0], input_image_size[1], 1)
        MASKS[j] = mask
    except:
        print(file)
        continue

images_train, images_test, masks_train, masks_test = train_test_split(
    IMAGES, MASKS, test_size=0.4, random_state=42)


import tensorflow as tf
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.models import Model


def FastRCNN_Segmentation(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    # Base model: VGG16
    base_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_tensor=inputs)

    # Encoder (downsampling path)
    conv1 = base_model.get_layer('block1_conv2').output
    conv2 = base_model.get_layer('block2_conv2').output
    conv3 = base_model.get_layer('block3_conv3').output
    conv4 = base_model.get_layer('block4_conv3').output
    conv5 = base_model.get_layer('block5_conv3').output

    # Decoder (upsampling path)
    up_conv5 = UpSampling2D((2, 2))(conv5)
    up_conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(up_conv5)
    up_conv4 = concatenate([up_conv5, conv4], axis=-1)
    up_conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(up_conv4)
    up_conv4 = UpSampling2D((2, 2))(up_conv4)
    up_conv3 = concatenate([up_conv4, conv3], axis=-1)
    up_conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(up_conv3)
    up_conv3 = UpSampling2D((2, 2))(up_conv3)
    up_conv2 = concatenate([up_conv3, conv2], axis=-1)
    up_conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(up_conv2)
    up_conv2 = UpSampling2D((2, 2))(up_conv2)
    up_conv1 = concatenate([up_conv2, conv1], axis=-1)
    up_conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(up_conv1)

    # Output segmentation mask
    segmentation_output = Conv2D(num_classes, (1, 1), activation='softmax', name='segmentation')(up_conv1)

    # Define the model
    model = Model(inputs=inputs, outputs=segmentation_output, name="FastRCNNSegmentation")

    return model


# Define input shape and number of classes
input_shape = (144, 144, 3)  # Example input shape
num_classes = 10  # Example number of classes

# Initialize the model
model = FastRCNN_Segmentation(input_shape, num_classes)

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])

# Print model summary
model.summary()

# Dummy data for testing the model
dummy_images = np.random.random((1, 144, 144, 3)).astype(np.float32)
dummy_labels = np.random.randint(0, num_classes, (1, 144, 144)).astype(np.int32)

# Check the output shapes
# print(f"Dummy images shape: {dummy_images.shape}")
# print(f"Dummy labels shape: {dummy_labels.shape}")
#
# # Predict to verify shape
# predictions = model.predict(dummy_images)
# print(f"Prediction shape: {predictions.shape}")

# Perform a single training step to check for errors
model.train_on_batch(dummy_images, dummy_labels)
# Example usage
input_image_size = (144, 144)
num_classes = 2  # For binary segmentation, change it according to your task
fast_rcnn_segmentation_model = FastRCNN_Segmentation(input_shape=(input_image_size[0], input_image_size[1], 3), num_classes=num_classes)
fast_rcnn_segmentation_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
fast_rcnn_segmentation_model.summary()


# Early Stopping
# early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
start_time = time.time()
history = fast_rcnn_segmentation_model.fit(
    images_train, masks_train,
    validation_split=0.2,
    batch_size=4,
    epochs=25#,
    #callbacks=[early_stopping]
)
end_time = time.time()

# Calculate training time in minutes
training_time_minutes = (end_time - start_time) / 60

# Evaluate the model
test_loss, test_accuracy = fast_rcnn_segmentation_model.evaluate(images_test, masks_test, verbose=0)

# Prediction
y_pred = fast_rcnn_segmentation_model.predict(images_test)

# Calculate precision, recall, and F1 score
precision = precision_score(np.ravel(masks_test), np.ravel(y_pred) > 0.5)
recall = recall_score(np.ravel(masks_test), np.ravel(y_pred) > 0.5)
f1 = f1_score(np.ravel(masks_test), np.ravel(y_pred) > 0.5)

# Specify the filename
filename = "fast_rcnn_model 25 Epochs.txt"

# Create the file if it doesn't exist
if not os.path.exists(filename):
    with open(filename, 'w') as f:
        f.write("")

# Save results to a text file and plots as a single image
with open(filename, 'a') as f:
    f.write(f"Number of Epochs: {len(history.epoch)}\n")
    f.write(f"Training Time (minutes): {training_time_minutes}\n")
    f.write(f"Test Loss: {test_loss}\n")
    f.write(f"Test Accuracy: {test_accuracy}\n")
    f.write(f"Precision: {precision}\n")
    f.write(f"Recall: {recall}\n")
    f.write(f"F1 Score: {f1}\n")

# Plot loss and accuracy
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Curves')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Curves')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()

# Save plots and text file with the same name
file_basename = os.path.splitext(filename)[0]

plt.savefig(f"{file_basename}.png")
plt.show()