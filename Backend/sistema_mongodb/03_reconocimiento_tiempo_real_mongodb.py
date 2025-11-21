import cv2
import numpy as np
import tensorflow as tf
import os
from pymongo import MongoClient
import time

# --- CONFIGURACIÓN ---
MONGO_URI = "mongodb+srv://lithubprogramadores_db_user:NlejQAZ9OuLJnv55@ai.xr4bewc.mongodb.net/?appName=Ai"
DB_NAME = "face_recognition_system_Scripts"
COLLECTION_NAME = "face_encodings"
MODELO_PATH = "modelo_rostros_mongodb.h5"
CLASES_PATH = "clases_mongodb.npy"
# ---------------------

# Opcional: Silenciar logs de TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# --- CONEXIÓN A MONGO ---
try:
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    # Verificar conexión
    client.admin.command('ping')
    print("Conexión a MongoDB exitosa")
except Exception as e:
    print(f"Error conectando a MongoDB: {str(e)}")
    client = None
    db = None
    collection = None

def extraer_caracteristicas_rostro(imagen_rostro):
    """
    MISMÍSIMA función usada en el entrenamiento y captura.
    Es vital que sea idéntica paso a paso.
    """
    try:
        # 1. Redimensionar a 64x64 (Igual que en el entrenamiento)
        rostro_redimensionado = cv2.resize(imagen_rostro, (64, 64))
        
        # 2. Convertir a grises
        if len(rostro_redimensionado.shape) == 3:
            gray = cv2.cvtColor(rostro_redimensionado, cv2.COLOR_BGR2GRAY)
        else:
            gray = rostro_redimensionado
        
        # Asegurar tipo de dato
        gray = gray.astype(np.uint8)
        
        # 3. Preprocesamiento (Igual que en entrenamiento)
        gray = cv2.equalizeHist(gray)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # 4. Extraer características por regiones
        h, w = gray.shape
        caracteristicas = []
        
        # Dividir en 8x8 regiones
        for i in range(0, h, 8):
            for j in range(0, w, 8):
                region = gray[i:i+8, j:j+8]
                if region.size > 0:
                    caracteristicas.append(np.mean(region))
                    caracteristicas.append(np.std(region))
        
        # Agregar estadísticas globales
        caracteristicas.append(np.mean(gray))
        caracteristicas.append(np.std(gray))
        caracteristicas.append(np.median(gray))
        
        # Rellenar hasta 128 características
        while len(caracteristicas) < 128:
            caracteristicas.append(0.0)
        
        # Recortar si sobran
        caracteristicas = caracteristicas[:128]
        
        # Normalizar vector
        caracteristicas = np.array(caracteristicas, dtype=np.float32)
        norm = np.linalg.norm(caracteristicas)
        if norm > 0:
            caracteristicas = caracteristicas / norm
        
        return caracteristicas
        
    except Exception as e:
        print(f"Error extrayendo características: {e}")
        return None

def get_database_info():
    """Obtiene información de las personas en la base de datos"""
    if collection is None:
        return []
    
    try:
        persons = collection.find({}, {'person_name': 1, 'total_encodings': 1, 'capture_date': 1})
        return list(persons)
    except Exception as e:
        print(f"Error consultando base de datos: {str(e)}")
        return []

def save_recognition_log(person_detected, confidence, timestamp):
    """Guarda log de reconocimientos en MongoDB"""
    if db is None:
        return
    
    try:
        log_collection = db["recognition_logs"]
        
        log_entry = {
            'person_detected': person_detected,
            'confidence': float(confidence),
            'timestamp': timestamp,
            'recognition_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'model_used': MODELO_PATH
        }
        
        log_collection.insert_one(log_entry)
        
    except Exception as e:
        pass  # No mostrar errores de logging para no interrumpir el reconocimiento

def main():
    print("=" * 60)
    print("RECONOCIMIENTO FACIAL EN TIEMPO REAL")
    print("Paso 3: Reconocimiento usando modelo entrenado con MongoDB")
    print("=" * 60)

    # 1. Verificar información de la base de datos
    print("Información de la base de datos:")
    db_persons = get_database_info()
    if db_persons:
        for person in db_persons:
            print(f"   • {person['person_name']}: {person.get('total_encodings', 'N/A')} huellas")
    else:
        print("   No se encontraron personas en la base de datos")

    # 2. Cargar Modelo y Clases
    print(f"\nCargando modelo entrenado...")
    try:
        if not os.path.exists(MODELO_PATH) or not os.path.exists(CLASES_PATH):
            print(f"No se encuentran los archivos del modelo:")
            print(f"   • {MODELO_PATH}")
            print(f"   • {CLASES_PATH}")
            print(f"\nSolución: Ejecuta primero python 02_entrenar_modelo_mongodb.py")
            return

        model = tf.keras.models.load_model(MODELO_PATH)
        clases = np.load(CLASES_PATH, allow_pickle=True)
        print(f"Modelo cargado exitosamente")
        print(f"   • Clases disponibles: {list(clases)}")
        print(f"   • Total de clases: {len(clases)}")
        
    except Exception as e:
        print(f"Error cargando modelo: {str(e)}")
        return

    # 3. Cargar Detector Facial
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        print("No se pudo cargar el detector de rostros de OpenCV")
        return

    # 4. Iniciar Cámara
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se puede abrir la cámara")
        return

    # Configurar cámara para mejor calidad
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print(f"\nCámara iniciada exitosamente")
    print(f"Instrucciones:")
    print(f"   • Mantén buena iluminación")
    print(f"   • Mira directamente a la cámara")
    print(f"   • Presiona 'Q' para salir")
    print(f"   • Presiona 'S' para ver estadísticas")
    
    # Variables para estadísticas
    total_detections = 0
    recognition_counts = {name: 0 for name in clases}
    recognition_counts['Desconocido'] = 0
    
    print(f"\nINICIANDO RECONOCIMIENTO EN TIEMPO REAL...")
    print(f"   (Los reconocimientos se guardan automáticamente en MongoDB)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Copia del frame para dibujar
        display_frame = frame.copy()
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(80, 80)  # Tamaño mínimo consistente con la captura
        )

        for (x, y, w, h) in faces:
            # Extraer ROI (Region of Interest)
            face_roi = frame[y:y+h, x:x+w]
            
            # Obtener vector de características
            features = extraer_caracteristicas_rostro(face_roi)

            if features is not None:
                # Hacer predicción
                features_batch = features.reshape(1, -1)
                prediction = model.predict(features_batch, verbose=0)[0]
                
                # Obtener el índice con mayor probabilidad
                idx_max = np.argmax(prediction)
                confidence = prediction[idx_max]
                nombre_detectado = clases[idx_max]

                # Lógica de umbrales de confianza
                color = (0, 0, 255)  # Rojo por defecto
                
                if confidence > 0.80:
                    color = (0, 255, 0)  # Verde
                    label = f"{nombre_detectado}"
                    person_result = nombre_detectado
                elif confidence > 0.50:
                    color = (0, 255, 255)  # Amarillo
                    label = f"Posible {nombre_detectado}?"
                    person_result = f"Posible_{nombre_detectado}"
                else:
                    label = "Desconocido"
                    person_result = "Desconocido"

                # Actualizar estadísticas
                total_detections += 1
                if person_result in recognition_counts:
                    recognition_counts[person_result] += 1
                else:
                    recognition_counts[person_result] = 1
                
                # Guardar en log de MongoDB (cada 30 frames para no sobrecargar)
                if total_detections % 30 == 0:
                    save_recognition_log(person_result, confidence, time.time())

                # Dibujar resultado
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(display_frame, f"{label} ({confidence:.1%})", 
                           (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Mostrar información en pantalla
        cv2.putText(display_frame, f"Detecciones totales: {total_detections}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_frame, "Presiona 'Q' para salir, 'S' para stats", 
                   (10, display_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("Reconocimiento Facial - MongoDB System", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Mostrar estadísticas
            print(f"\nESTADÍSTICAS ACTUALES:")
            print(f"   • Detecciones totales: {total_detections}")
            for person, count in recognition_counts.items():
                if count > 0:
                    percentage = (count / total_detections) * 100
                    print(f"   • {person}: {count} ({percentage:.1f}%)")

    cap.release()
    cv2.destroyAllWindows()
    
    # Resumen final
    print(f"\nSESIÓN DE RECONOCIMIENTO TERMINADA")
    print(f"   • Detecciones totales: {total_detections}")
    print(f"   • Personas detectadas: {len([p for p, c in recognition_counts.items() if c > 0])}")
    
    if total_detections > 0:
        print(f"\nDistribución de reconocimientos:")
        for person, count in recognition_counts.items():
            if count > 0:
                percentage = (count / total_detections) * 100
                print(f"     {person}: {count} ({percentage:.1f}%)")
    
    print(f"\nLos logs de reconocimiento han sido guardados en MongoDB")
    print(f"   • Colección: {DB_NAME}.recognition_logs")

if __name__ == "__main__":
    main()