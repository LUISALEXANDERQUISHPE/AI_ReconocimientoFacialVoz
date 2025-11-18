import cv2

def main():
    cap = cv2.VideoCapture(0)  # 0 = webcam principal

    if not cap.isOpened():
        print("No se pudo abrir la cámara")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se pudo leer el frame")
            break

        cv2.imshow("Camara - Presiona Q para salir", frame)

        # Si presionas 'q', sales
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
