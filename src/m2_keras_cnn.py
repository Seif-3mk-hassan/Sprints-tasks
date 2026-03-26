import numpy as np
from datasets import load_dataset
from PIL import Image

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers



dataset = load_dataset("blanchon/EuroSAT_RGB")

print(dataset)
print("Train samples:", len(dataset["train"]))
print("Validation samples:", len(dataset["validation"]))
print("Test samples:", len(dataset["test"]))


sample = dataset["train"][0]
print("Image:", sample["image"])
print("Label:", sample["label"])
print("Image size:", sample["image"].size)



def extract_images_labels(split):
    """
    Convert a Hugging Face split into NumPy arrays.
    - Images: float32 arrays normalized to [0, 1]
    - Labels: int arrays
    """
    images = []
    labels = []

    for sample in split:
        # Convert PIL Image → NumPy array (H, W, C) uint8
        img_array = np.array(sample["image"], dtype=np.float32)

        img_array /= 255.0

        images.append(img_array)
        labels.append(sample["label"])

    images = np.array(images)   # Shape: (N, 64, 64, 3)
    labels = np.array(labels)   # Shape: (N,)
    return images, labels

def keras_train_model():
    # ─── Build the CNN ───
    keras_model = keras.Sequential([
        layers.Input(shape=(64, 64, 3)),

        # Block 1
        layers.Conv2D(32, kernel_size=3, activation="relu"),
        layers.MaxPooling2D(pool_size=2),

        # Block 2
        layers.Conv2D(64, kernel_size=3, activation="relu"),
        layers.MaxPooling2D(pool_size=2),

        # Classifier head
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ])

    keras_model.summary()

    # ─── Compile ───
    keras_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # ─── Train ───
    EPOCHS = 10
    BATCH_SIZE = 32

    history = keras_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
    )

    # ─── Evaluate on test set ───
    test_loss, test_acc = keras_model.evaluate(X_test, y_test, verbose=0)
    print(f"\n[Keras] Test Loss: {test_loss:.4f}  |  Test Accuracy: {test_acc:.4f}")

    # ─── Save weights ───
    keras_model.save_weights("keras_cnn.weights.h5")
    print("Keras weights saved → keras_cnn.weights.h5")


if '__main__' == __name__:
    # Extract all three splits
    print("Processing train split...")
    X_train, y_train = extract_images_labels(dataset["train"])

    print("Processing validation split...")
    X_val, y_val = extract_images_labels(dataset["validation"])

    print("Processing test split...")
    X_test, y_test = extract_images_labels(dataset["test"])

    print(f"Train:      {X_train.shape}, Labels: {y_train.shape}")
    print(f"Validation: {X_val.shape},  Labels: {y_val.shape}")
    print(f"Test:       {X_test.shape},  Labels: {y_test.shape}")

    keras_train_model()


