Question 1: Generate Images using DCGAN
1.1. Implement DCGAN to generate images from noise
We'll start by implementing a DCGAN to generate images from noise using the CIFAR-10 dataset. The following steps outline the process:

Data Augmentation Function for GAN Training
Simple Discriminator Model
Generator Model
Minimax Loss Function
Complete GAN Model
python
Copy code
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Conv2DTranspose, Flatten, Reshape, LeakyReLU, BatchNormalization
from tensorflow.keras.models import Sequential
import numpy as np
import matplotlib.pyplot as plt

# Data Augmentation Function
def augment_data(images, rotation_range=30, flip_horizontal=True, flip_vertical=False):
    augmented_images = []
    for img in images:
        if flip_horizontal and np.random.rand() > 0.5:
            img = np.fliplr(img)
        if flip_vertical and np.random.rand() > 0.5:
            img = np.flipud(img)
        angle = np.random.uniform(-rotation_range, rotation_range)
        img = tf.keras.preprocessing.image.apply_affine_transform(img, theta=angle)
        augmented_images.append(img)
    return np.array(augmented_images)

# Load CIFAR-10 dataset
(x_train, _), (_, _) = tf.keras.datasets.cifar10.load_data()
x_train = (x_train.astype(np.float32) - 127.5) / 127.5  # Normalize to [-1, 1]
x_train = augment_data(x_train)

# Define the Discriminator Model
def build_discriminator(img_shape):
    model = Sequential([
        Conv2D(64, kernel_size=4, strides=2, input_shape=img_shape, padding="same"),
        LeakyReLU(alpha=0.2),
        Conv2D(128, kernel_size=4, strides=2, padding="same"),
        BatchNormalization(),
        LeakyReLU(alpha=0.2),
        Flatten(),
        Dense(1, activation='sigmoid')
    ])
    return model

# Define the Generator Model
def build_generator(latent_dim):
    model = Sequential([
        Dense(8*8*256, input_dim=latent_dim),
        Reshape((8, 8, 256)),
        BatchNormalization(),
        LeakyReLU(alpha=0.2),
        Conv2DTranspose(128, kernel_size=4, strides=2, padding="same"),
        BatchNormalization(),
        LeakyReLU(alpha=0.2),
        Conv2DTranspose(64, kernel_size=4, strides=2, padding="same"),
        BatchNormalization(),
        LeakyReLU(alpha=0.2),
        Conv2DTranspose(3, kernel_size=4, strides=1, padding="same", activation='tanh')
    ])
    return model

# Minimax Loss Function
def gan_loss(real_output, fake_output):
    real_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(real_output), real_output)
    fake_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

# Compile the GAN Model
latent_dim = 100
img_shape = (32, 32, 3)

discriminator = build_discriminator(img_shape)
discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

generator = build_generator(latent_dim)
gan_input = tf.keras.layers.Input(shape=(latent_dim,))
gan_output = discriminator(generator(gan_input))

gan = tf.keras.models.Model(gan_input, gan_output)
gan.compile(optimizer='adam', loss='binary_crossentropy')

# Training the GAN
def train_gan(epochs, batch_size, interval):
    for epoch in range(epochs):
        idx = np.random.randint(0, x_train.shape[0], batch_size)
        real_images = x_train[idx]

        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        fake_images = generator.predict(noise)

        real_labels = np.ones((batch_size, 1))
        fake_labels = np.zeros((batch_size, 1))

        d_loss_real = discriminator.train_on_batch(real_images, real_labels)
        d_loss_fake = discriminator.train_on_batch(fake_images, fake_labels)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        valid_y = np.ones((batch_size, 1))
        g_loss = gan.train_on_batch(noise, valid_y)

        if epoch % interval == 0:
            print(f"{epoch} [D loss: {d_loss[0]} | D acc.: {100*d_loss[1]}] [G loss: {g_loss}]")
            save_images(epoch)

def save_images(epoch, samples=10):
    noise = np.random.normal(0, 1, (samples, latent_dim))
    gen_images = generator.predict(noise)
    gen_images = 0.5 * gen_images + 0.5

    fig, axs = plt.subplots(1, samples, figsize=(20, 4))
    for i in range(samples):
        axs[i].imshow(gen_images[i])
        axs[i].axis('off')
    plt.show()

train_gan(epochs=10000, batch_size=64, interval=1000)
Question 2: Fine-Tuning ResNet50 on CIFAR-10
2.1. ResNet50 with Custom Classification Layer
python
Copy code
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Normalize data
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Load pre-trained ResNet50 without the top layer
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

# Add custom top layers
x = Flatten()(base_model.output)
x = Dense(512, activation='relu')(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(10, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))

# Plotting accuracies over epochs
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
2.2. Different Training Strategies with ResNet50
For parts B and C:

python
Copy code
# B. Fine-tuning only the custom layers (Freeze ResNet50 layers)
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history_b = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))

# Plotting accuracies over epochs for B
plt.plot(history_b.history['accuracy'], label='Train Accuracy (Frozen)')
plt.plot(history_b.history['val_accuracy'], label='Test Accuracy (Frozen)')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# C. Fine-tuning all layers
for layer in base_model.layers:
    layer.trainable = True

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history_c = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))

# Plotting accuracies over epochs for C
plt.plot(history_c.history['accuracy'], label='Train Accuracy (Fine-tuned)')
plt.plot(history_c.history['val_accuracy'], label='Test Accuracy (Fine-tuned)')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
Question 3: Implementing GAN from Scratch using Keras
3.1. Load CelebA Dataset and Preprocess
python
Copy code
import pandas as pd
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load CelebA dataset
def load_celeba(data_dir, img_size=(64, 64)):
    images = []
    img_dir = os.path.join(data_dir, 'img_align_celeba')
    for img_name in os.listdir(img_dir):
        img_path = os.path.join(img_dir, img_name)
        img = load_img(img_path, target_size=img_size)
        img = img_to_array(img)
        images.append(img)
    return np.array(images)

data_dir = 'path_to_celeba_dataset'
celeba_images = load_celeba(data_dir)
cele








