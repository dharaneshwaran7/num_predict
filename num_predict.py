import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

def preprocess_image(image):
    image = image.resize((28, 28))
    image = image.convert('L')  # Convert to grayscale
    image = np.array(image)
    image = image / 255.0  # Normalize
    return image

def predict_digit():
    global image
    user_image = image  # Get the drawn image from the canvas
    prediction = model.predict(np.expand_dims(preprocess_image(user_image), axis=0))
    predicted_digit_label.config(text="Predicted digit: " + str(np.argmax(prediction)))

def paint(event):
    x1, y1 = (event.x - 10), (event.y - 10)
    x2, y2 = (event.x + 10), (event.y + 10)
    canvas.create_oval(x1, y1, x2, y2, fill="black")
    draw.line([x1, y1, x2, y2], fill="black", width=20)

def clear():
    canvas.delete("all")
    draw.rectangle([0, 0, 280, 280], fill="white")
    predicted_digit_label.config(text="Predicted digit: ")

# Load and preprocess the dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Build the model
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),    # Flatten the input images
    layers.Dense(128, activation='relu'),    # Hidden layer with 128 neurons and ReLU activation
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.2),                     # Dropout layer to prevent overfitting
    layers.Dense(10, activation='softmax')   # Output layer with 10 neurons (one for each class) and softmax activation
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5)

# Setup user interface
root = tk.Tk()
root.title("Handwritten Digit Recognition")

canvas = tk.Canvas(root, width=280, height=280, bg="white")
canvas.pack()

image = Image.new("RGB", (280, 280), "white")
draw = ImageDraw.Draw(image)

canvas.bind("<B1-Motion>", paint)

clear_button = tk.Button(root, text="Clear", command=clear)
clear_button.pack()

predict_button = tk.Button(root, text="Predict", command=predict_digit)
predict_button.pack()

predicted_digit_label = tk.Label(root, text="Predicted digit: ")
predicted_digit_label.pack()

root.mainloop()
