import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
import numpy as np
import matplotlib.pyplot as plt

# Load pre-trained ResNet50 model
model = ResNet50(weights='imagenet')


def preprocess_image(image_path):
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    return image


def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = tf.sign(data_grad)
    perturbed_image = image + epsilon * sign_data_grad
    perturbed_image = tf.clip_by_value(perturbed_image, -1, 1)
    return perturbed_image


def generate_adversarial_example(image, epsilon):
    image_tensor = tf.convert_to_tensor(image)
    with tf.GradientTape() as tape:
        tape.watch(image_tensor)
        prediction = model(image_tensor)
        loss = tf.keras.losses.categorical_crossentropy(prediction, prediction)

    gradient = tape.gradient(loss, image_tensor)
    perturbed_image = fgsm_attack(image_tensor, epsilon, gradient)
    return perturbed_image


# Load and preprocess the image
image_path = "../data/pizza_2.jpg"
original_image = preprocess_image(image_path)

# Generate adversarial example
epsilon = 0.01  # You can adjust this value
adversarial_image = generate_adversarial_example(original_image, epsilon)

# Make predictions
original_pred = model.predict(original_image)
adversarial_pred = model.predict(adversarial_image)

# Decode predictions
original_label = decode_predictions(original_pred)[0][0]
adversarial_label = decode_predictions(adversarial_pred)[0][0]

# Display results
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(tf.keras.preprocessing.image.array_to_img(original_image[0]))
plt.title(f"Original: {original_label[1]}\nConfidence: {original_label[2]:.2f}")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(tf.keras.preprocessing.image.array_to_img(adversarial_image[0].numpy()))
plt.title(f"Adversarial: {adversarial_label[1]}\nConfidence: {adversarial_label[2]:.2f}")
plt.axis('off')

plt.tight_layout()
plt.show()

print(f"Original prediction: {original_label[1]} ({original_label[2]:.2f})")
print(f"Adversarial prediction: {adversarial_label[1]} ({adversarial_label[2]:.2f})")
