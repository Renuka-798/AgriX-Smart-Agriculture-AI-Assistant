import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Load trained model
model = tf.keras.models.load_model(
    "trained_models/disease_model.keras"
)


# Automatically get class names
dataset_path = "datasets/plantVillage"

datagen = ImageDataGenerator(
    rescale=1./255
)

generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(128, 128),
    batch_size=32,
    class_mode="categorical",
    shuffle=False
)

class_names = list(
    generator.class_indices.keys()
)


def predict_leaf_disease(img_path):

    img = image.load_img(
        img_path,
        target_size=(128, 128)
    )

    img_array = image.img_to_array(img)

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    img_array = img_array / 255.0

    prediction = model.predict(
        img_array
    )

    predicted_class = np.argmax(
        prediction
    )

    confidence = np.max(
        prediction
    ) * 100

    disease_name = class_names[
        predicted_class
    ]

    # Clean text
    disease_name = disease_name.replace(
        "_",
        " "
    )

    return disease_name, confidence