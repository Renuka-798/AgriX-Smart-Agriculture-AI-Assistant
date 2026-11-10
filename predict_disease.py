
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image


# Load the trained disease detection model
model = tf.keras.models.load_model(
    "trained_models/disease_model.keras"
)


# Class names in the exact alphabetical order
# used by flow_from_directory() during training
class_names = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "PlantVillage",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy"
]


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
        img_array,
        verbose=0
    )

    predicted_class = np.argmax(prediction)

    confidence = float(np.max(prediction)) * 100

    disease_name = class_names[predicted_class]

    disease_name = disease_name.replace("_", " ")

    return disease_name, confidence
