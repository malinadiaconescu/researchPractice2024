import os
import numpy as np
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import time

# Update the paths to your directories
local_images_dir = r'/frames'
local_masks_dir = r'/masks'

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

# Model
def SimpleUnet(input_shape):
    inputs = tf.keras.layers.Input(input_shape)

    # Encoder
    conv1 = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    # Bottleneck
    conv2 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(pool1)

    # Decoder
    up3 = tf.keras.layers.Conv2DTranspose(32, 2, strides=(2, 2), padding='same')(conv2)
    up3 = tf.keras.layers.concatenate([up3, conv1], axis=3)
    conv3 = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(up3)

    outputs = tf.keras.layers.Conv2D(1, 1, activation='sigmoid')(conv3)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

simple_unet_model = SimpleUnet((input_image_size[0], input_image_size[1], 3))
simple_unet_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Early Stopping
# early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
start_time = time.time()
history = simple_unet_model.fit(
    images_train, masks_train,
    validation_split=0.2,
    batch_size=4,
    epochs=50#,
    #callbacks=[early_stopping]
)
end_time = time.time()

# Calculate training time in minutes
training_time_minutes = (end_time - start_time) / 60

# Evaluate the model
test_loss, test_accuracy = simple_unet_model.evaluate(images_test, masks_test, verbose=0)

# Prediction
y_pred = simple_unet_model.predict(images_test)

# Calculate precision, recall, and F1 score
precision = precision_score(np.ravel(masks_test), np.ravel(y_pred) > 0.5)
recall = recall_score(np.ravel(masks_test), np.ravel(y_pred) > 0.5)
f1 = f1_score(np.ravel(masks_test), np.ravel(y_pred) > 0.5)

# Specify the filename
filename = "Unet 50 Epochs.txt"

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
