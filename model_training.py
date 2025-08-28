# %%
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")   # Non-GUI backend (saves plots to file instead of showing)
import matplotlib.pyplot as plt
import cv2

from tensorflow.keras import applications, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense

# %% [markdown]
# Count training and validation images

files_train, files_validation = 0, 0

train_folder = 'train_data/train'
test_folder = 'train_data/test'

for sub_folder in os.listdir(train_folder):
    _, _, files = next(os.walk(os.path.join(train_folder, sub_folder)))
    files_train += len(files)

for sub_folder in os.listdir(test_folder):
    _, _, files = next(os.walk(os.path.join(test_folder, sub_folder)))
    files_validation += len(files)

print(f"Training samples: {files_train}, Validation samples: {files_validation}")

# %% [markdown]
# Set key parameters

img_width, img_height = 48, 48
train_data_dir = 'train_data/train'
validation_data_dir = 'train_data/test'
nb_train_sample = files_train
nb_validation_sample = files_validation
batch_size = 32
epochs = 15
num_classes = 2

# %% [markdown]
# Build model on top of a pretrained VGG16

base_model = applications.VGG16(
    weights="imagenet",
    include_top=False,
    input_shape=(img_width, img_height, 3)
)

# Freeze early layers
for layer in base_model.layers[:10]:
    layer.trainable = False

# Add custom head
x = base_model.output
x = Flatten()(x)
predictions = Dense(num_classes, activation="softmax")(x)

model_final = Model(inputs=base_model.input, outputs=predictions)

model_final.compile(
    loss="categorical_crossentropy",
    optimizer=optimizers.SGD(learning_rate=0.0001, momentum=0.9),
    metrics=["accuracy"]
)

# %% [markdown]
# Data Augmentation

train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    horizontal_flip=True,
    fill_mode="nearest",
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5
)

test_datagen = ImageDataGenerator(
    rescale=1.0/255
)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="categorical"
)

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="categorical"
)

# %% [markdown]
# Train the model

history = model_final.fit(
    train_generator,
    steps_per_epoch=nb_train_sample // batch_size,
    validation_data=validation_generator,
    validation_steps=nb_validation_sample // batch_size,
    epochs=epochs
)

# %% [markdown]
# Plot Accuracy

plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("Model Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epochs")
plt.legend(["Train", "Validation"], loc="upper left")
plt.savefig("accuracy_plot.png")
plt.close()



# %% [markdown]
# Plot Loss

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Model Loss")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend(["Train", "Validation"], loc="upper left")
plt.savefig("loss_plot.png")
plt.close()
# %% [markdown]
# Save model

model_final.save("model_final.h5")

# %% [markdown]
# Class Dictionary

class_dictionary = {0: "no_car", 1: "car"}
print("Class mapping:", class_dictionary)

# %% [markdown]
# Prediction Function

def make_prediction(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (48, 48))
    img = image / 255.0
    img = np.expand_dims(img, axis=0)  # (1, 48, 48, 3)
    
    class_predicted = model_final.predict(img, verbose=0)
    intId = np.argmax(class_predicted[0])
    label = class_dictionary[intId]
    return label

# %% Test predictions
print(make_prediction("roi_1.png"))
print(make_prediction("spot169.jpg"))
