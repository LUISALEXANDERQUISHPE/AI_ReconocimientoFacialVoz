import cv2
import numpy as np
import face_recognition
import tensorflow as tf

MODELO_PATH = "modelo_rostros.h5"
CLASES_PATH = "clases.npy"


def main():
    print("Cargando modelo...")
    model = tf.keras.models.load_model(MODELO_PATH)
    clases = np.load(CLASES_PATH, allow_pickle=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se pudo abrir la cámara")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se pudo leer el frame")
            break

        # Copias para procesar y mostrar
        rgb = frame[:, :, ::-1]

        # Detectar rostros
        face_locations = face_recognition.face_locations(rgb)

        for (top, right, bottom, left) in face_locations:
            # Obtener encoding
            encoding = face_recognition.face_encodings(rgb, [(top, right, bottom, left)])
            if len(encoding) == 0:
                continue

            encoding = encoding[0].astype("float32")
            pred = model.predict(np.array([encoding]), verbose=0)[0]
            idx = np.argmax(pred)
            prob = float(pred[idx])
            name = clases[idx]

            # Dibujar recuadro y nombre
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            text = f"{name} ({prob*100:.1f}%)"
            cv2.putText(frame, text, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Reconocimiento de rostros - Q para salir", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
