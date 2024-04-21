import os
import numpy as np
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import time

# Update the paths to your directories
local_images_dir = r'C:\Users\Adrian\Desktop\medical-image-segmentation\frames'
local_masks_dir = r'C:\Users\Adrian\Desktop\medical-image-segmentation\frames'

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

def conv_block(input, num_filters):
    conv = tf.keras.layers.Conv2D(num_filters, 3, padding="same")(input)
    conv = tf.keras.layers.BatchNormalization()(conv)
    conv = tf.keras.layers.Activation("relu")(conv)
    conv = tf.keras.layers.Conv2D(num_filters, 3, padding="same")(conv)
    conv = tf.keras.layers.BatchNormalization()(conv)
    conv = tf.keras.layers.Activation("relu")(conv)
    return conv

def encoder_block(input, num_filters):
    skip = conv_block(input, num_filters)
    pool = tf.keras.layers.MaxPool2D((2,2))(skip)
    return skip, pool

def decoder_block(input, skip, num_filters):
    up_conv = tf.keras.layers.Conv2DTranspose(num_filters, (2,2), strides=2, padding="same")(input)
    conv = tf.keras.layers.Concatenate()([up_conv, skip])
    conv = conv_block(conv, num_filters)
    return conv

def FCN(input_shape):
    inputs = tf.keras.layers.Input(input_shape)

    skip1, pool1 = encoder_block(inputs, 64)
    skip2, pool2 = encoder_block(pool1, 128)
    skip3, pool3 = encoder_block(pool2, 256)
    skip4, pool4 = encoder_block(pool3, 512)

    bridge = conv_block(pool4, 1024)

    decode1 = decoder_block(bridge, skip4, 512)
    decode2 = decoder_block(decode1, skip3, 256)
    decode3 = decoder_block(decode2, skip2, 128)
    decode4 = decoder_block(decode3, skip1, 64)

    # Replace the dense layer with a convolutional layer
    outputs = tf.keras.layers.Conv2D(1, 1, padding="same", activation="sigmoid")(decode4)

    model = tf.keras.models.Model(inputs, outputs, name="FCN")
    return model

fcn_model = FCN(((input_image_size[0], input_image_size[1], 3)))
fcn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
fcn_model.summary()

# Early Stopping
# early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
start_time = time.time()
history = fcn_model.fit(
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
test_loss, test_accuracy = fcn_model.evaluate(images_test, masks_test, verbose=0)

# Prediction
y_pred = fcn_model.predict(images_test)

# Calculate precision, recall, and F1 score
precision = precision_score(np.ravel(masks_test), np.ravel(y_pred) > 0.5)
recall = recall_score(np.ravel(masks_test), np.ravel(y_pred) > 0.5)
f1 = f1_score(np.ravel(masks_test), np.ravel(y_pred) > 0.5)

# Specify the filename
filename = "FCN 25 Epochs.txt"

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
