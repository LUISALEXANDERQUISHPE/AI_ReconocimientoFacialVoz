import cv2
import face_recognition
import os

# ------ CONFIGURACION ------
NUM_FOTOS = 6          # cuántas fotos por persona
DATASET_DIR = "dataset"  # carpeta base del dataset
# ---------------------------

def main():
    nombre = input("Escribe el nombre de la persona: ").strip()
    if not nombre:
        print("Nombre vacío, saliendo...")
        return

    # Crear carpeta dataset/<nombre>
    person_dir = os.path.join(DATASET_DIR, nombre)
    os.makedirs(person_dir, exist_ok=True)
    print(f"Las fotos se guardarán en: {person_dir}")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se pudo abrir la cámara")
        return

    contador = 0

    while contador < NUM_FOTOS:
        ret, frame = cap.read()
        if not ret:
            print("No se pudo leer el frame")
            break

        rgb = frame[:, :, ::-1]
        face_locations = face_recognition.face_locations(rgb)

        # Si detecta alguna cara, tomamos la primera
        if face_locations:
            (top, right, bottom, left) = face_locations[0]

            # Dibujar rectángulo para que veas qué se va a cortar
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        cv2.putText(frame, f"Fotos tomadas: {contador}/{NUM_FOTOS}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Captura dataset - ENTER = capturar, Q = salir", frame)

        key = cv2.waitKey(1) & 0xFF

        # ENTER para capturar
        if key == 13:  # tecla Enter
            if not face_locations:
                print("No se detectó rostro, intenta de nuevo.")
                continue

            (top, right, bottom, left) = face_locations[0]
            rostro = frame[top:bottom, left:right]

            filename = os.path.join(person_dir, f"{contador}.jpg")
            cv2.imwrite(filename, rostro)
            print(f"Guardada: {filename}")
            contador += 1

        # Q para salir
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Fin de la captura.")

if __name__ == "__main__":
    main()
