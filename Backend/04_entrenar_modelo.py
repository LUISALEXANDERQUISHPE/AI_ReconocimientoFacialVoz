import os
import numpy as np
import face_recognition

from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras import layers, models

# ---------------- CONFIGURACIÓN ----------------
DATASET_DIR = "dataset"
MODELO_PATH = "modelo_rostros.h5"
CLASES_PATH = "clases.npy"
# ------------------------------------------------


def cargar_encodings_y_labels():
    X = []
    y = []

    for person_name in os.listdir(DATASET_DIR):
        person_dir = os.path.join(DATASET_DIR, person_name)
        if not os.path.isdir(person_dir):
            continue

        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            # Cargar imagen
            image = face_recognition.load_image_file(img_path)
            boxes = face_recognition.face_locations(image)

            if len(boxes) == 0:
                print(f"[AVISO] No se detectó rostro en {img_path}, se omite.")
                continue

            encodings = face_recognition.face_encodings(image, boxes)
            if len(encodings) == 0:
                print(f"[AVISO] No se pudo obtener encoding en {img_path}, se omite.")
                continue

            encoding = encodings[0]
            X.append(encoding)
            y.append(person_name)

    X = np.array(X, dtype="float32")
    y = np.array(y)

    return X, y


def construir_modelo(num_clases):
    model = models.Sequential([
        layers.Input(shape=(128,)),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_clases, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def main():
    print("Cargando encodings desde el dataset...")
    X, y = cargar_encodings_y_labels()

    if len(X) == 0:
        print("No se encontraron datos. ¿Seguro que tienes imágenes en 'dataset/'?")
        return

    print(f"Total de muestras: {X.shape[0]}")
    print(f"Dimensión de cada encoding: {X.shape[1]}")

    # Codificar etiquetas de texto a números
    le = LabelEncoder()
    y_int = le.fit_transform(y)
    clases = le.classes_
    num_clases = len(clases)

    print(f"Clases encontradas: {list(clases)}")
    y_cat = tf.keras.utils.to_categorical(y_int, num_classes=num_clases)

    # Construir modelo
    model = construir_modelo(num_clases)

    # Entrenar
    history = model.fit(
        X, y_cat,
        epochs=50,
        batch_size=8,
        validation_split=0.2,
        verbose=1
    )

    # Guardar modelo y clases
    model.save(MODELO_PATH)
    np.save(CLASES_PATH, clases)

    print(f"Modelo guardado en {MODELO_PATH}")
    print(f"Clases guardadas en {CLASES_PATH}")


if __name__ == "__main__":
    main()
