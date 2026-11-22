"""
MNIST CNN Image Classification + Streamlit Application
Student: B. Abhishek
Dataset: MNIST

Part A: CNN Image Classification
Part B: Streamlit Application
"""

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import tensorflow as tf
from PIL import Image


# ============================================================
# PART A — CNN IMAGE CLASSIFICATION
# ============================================================

@st.cache_resource
def train_cnn():
    # 1. Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # 2. Preprocess images
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Add channel dimension: (28, 28) -> (28, 28, 1)
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    # 3. Build CNN model
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation="softmax")
    ])

    # 4. Compile model
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # 5. Train model
    history = model.fit(
        x_train,
        y_train,
        epochs=5,
        batch_size=128,
        validation_split=0.1,
        verbose=1
    )

    # 6. Evaluate on test data
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)

    # 7. Generate predictions
    probabilities = model.predict(x_test, verbose=0)
    predicted_labels = np.argmax(probabilities, axis=1)

    return (
        model,
        history,
        x_test,
        y_test,
        predicted_labels,
        test_loss,
        test_accuracy
    )


# Train the CNN once and cache it for the Streamlit session
(
    model,
    history,
    x_test,
    y_test,
    predicted_labels,
    test_loss,
    test_accuracy
) = train_cnn()


# ============================================================
# PART B — STREAMLIT APPLICATION
# ============================================================

st.set_page_config(
    page_title="MNIST Digit Recognizer",
    page_icon="🔢",
    layout="centered"
)

st.title("MNIST Handwritten Digit Recognizer")

st.write(
    "Upload an image of a handwritten digit from 0 to 9. "
    "The CNN model will predict the digit."
)

# Model performance
st.sidebar.header("Model Performance")
st.sidebar.metric("Test Accuracy", f"{test_accuracy * 100:.2f}%")
st.sidebar.write(f"Test Loss: {test_loss:.4f}")

st.subheader("Upload a Digit Image")

uploaded_file = st.file_uploader(
    "Choose a PNG, JPG, or JPEG image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    try:
        # Open image and convert to grayscale
        original_image = Image.open(uploaded_file).convert("L")

        # Display uploaded image
        st.image(original_image, caption="Uploaded Image", width=250)

        # Resize to MNIST dimensions
        processed_image = original_image.resize((28, 28))

        # Convert to NumPy array and normalize
        image_array = np.array(processed_image).astype("float32") / 255.0

        # MNIST uses white digits on a black background.
        # Invert images that appear to have a light background.
        if image_array.mean() > 0.5:
            image_array = 1.0 - image_array

        # Add batch and channel dimensions
        image_array = image_array.reshape(1, 28, 28, 1)

        # Predict digit
        prediction = model.predict(image_array, verbose=0)[0]
        predicted_digit = int(np.argmax(prediction))
        confidence = float(np.max(prediction) * 100)

        # Display result
        st.success(f"Predicted Digit: {predicted_digit}")
        st.write(f"Confidence: {confidence:.2f}%")

        # Display prediction probabilities
        st.subheader("Prediction Probabilities")
        st.bar_chart(
            {
                str(digit): float(probability * 100)
                for digit, probability in enumerate(prediction)
            }
        )

    except Exception as error:
        st.error(
            "Unable to process this image. Please upload a clear digit image."
        )
        st.caption(f"Error details: {error}")


# ============================================================
# CNN RESULTS / EVIDENCE
# ============================================================

with st.expander("View CNN Results"):
    st.write(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    st.write(f"Test Loss: {test_loss:.4f}")

    st.write("First 10 Actual vs Predicted Labels:")

    for i in range(10):
        st.write(
            f"Image {i}: Actual = {int(y_test[i])}, "
            f"Predicted = {int(predicted_labels[i])}"
        )

with st.expander("Training Curves"):
    fig1, ax1 = plt.subplots()
    ax1.plot(
        history.history["accuracy"],
        label="Training Accuracy"
    )
    ax1.plot(
        history.history["val_accuracy"],
        label="Validation Accuracy"
    )
    ax1.set_title("Training and Validation Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    st.pyplot(fig1)
    plt.close(fig1)

    fig2, ax2 = plt.subplots()
    ax2.plot(
        history.history["loss"],
        label="Training Loss"
    )
    ax2.plot(
        history.history["val_loss"],
        label="Validation Loss"
    )
    ax2.set_title("Training and Validation Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    st.pyplot(fig2)
    plt.close(fig2)

st.caption("CNN-based MNIST handwritten digit classification project")
