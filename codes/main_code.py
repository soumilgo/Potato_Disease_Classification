import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import models, layers
IMAGE_SIZE=256
BATCH_SIZE=32
CHANNELS= 3
dataset = tf.keras.preprocessing.image_dataset_from_directory("PlantVillage", shuffle=True, image_size=(IMAGE_SIZE, IMAGE_SIZE), batch_size = BATCH_SIZE)
class_names = dataset.class_names
#print(class_names)
# plt.figure(figsize=(20,20))
# for image, label in dataset.take(1):
#     for i in range(32):
#         plt.subplot(7,6,i+1)
#         plt.imshow(image[i].numpy().astype("uint8"))
#         plt.title(class_names[label[i]])
#         plt.axis('off')
# plt.show()

def get_dataset_partition(ds,train_split = 0.8, val_split=0.1, test_split=0.1, shuffle=True, shuffle_size=1000):
    ds_size = len(ds)
    if shuffle:
        ds.shuffle(shuffle_size, seed=12)
    train_size = int(train_split*ds_size)
    val_size = int(val_split*ds_size)
    train_ds  = ds.take(train_size)
    val_ds  = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)

    return train_ds , test_ds, val_ds

train_ds, test_ds, val_ds = get_dataset_partition(dataset)


train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE).shuffle(1000)
val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE).shuffle(1000)
test_ds = test_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE).shuffle(1000)

resize_and_rescale = tf.keras.Sequential([
    layers.Resizing(IMAGE_SIZE, IMAGE_SIZE),
    layers.Rescaling(1.0/255)
])

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("Horizontal_and_vertical"),
    layers.RandomRotation(0.2),
])

# print(len(train_ds))
input_shape= (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
n_classes=3

model = models.Sequential([
    resize_and_rescale,
    data_augmentation,
    layers.Conv2D(32, (3,3), activation='relu', input_shape = input_shape),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(n_classes, activation='softmax')
])
model.build(input_shape=input_shape)
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)
history= model.fit(train_ds, epochs=50, batch_size=BATCH_SIZE, verbose=1, validation_data=val_ds)
#scores = model.evaluate(test_ds)
def predict(model, img):
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array,0)
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100*(np.max(predictions[0])),2)
    return predicted_class, confidence

plt.figure(figsize=(15,15))
for images, labels in test_ds.take(1):
    for(i) in range(9):
        plt.subplot(3,3,i+1)
        plt.imshow(images[i].numpy().astype('uint8'))
        predicted_class, confidence = predict(model, images[i])
        actual_class = class_names[labels[i]]
        plt.title(f"Actual: {actual_class}, \n, Predicted: {predicted_class}, \n, Confidence: {confidence}")
        plt.axis('off')
plt.show()

#model.export(f"../models/{1}")
import os
model_versions = max([int(i) for i in os.listdir("../models")+[0]])+1
model.export(f"../models/{model_versions}")
# model.save("../models/4".keras)