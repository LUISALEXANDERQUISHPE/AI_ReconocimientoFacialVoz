import cv2
import numpy as np
import os
import time
from pathlib import Path
from pymongo import MongoClient
import json

# ------ CONFIGURACION ------
NUM_FOTOS = 100         # cuántas fotos extraer del video
DURACION_VIDEO = 40     # duración del video en segundos
DATASET_DIR = "dataset_raw"  # carpeta base del dataset crudo
FACE_SIZE_MIN = (80, 80)     # tamaño mínimo de rostro a detectar
QUALITY_THRESHOLD = 0.3      # umbral de calidad para descartar imágenes borrosas
MONGO_URI = "mongodb+srv://lithubprogramadores_db_user:NlejQAZ9OuLJnv55@ai.xr4bewc.mongodb.net/?appName=Ai"
DB_NAME = "face_recognition_system_Scripts"
COLLECTION_NAME = "face_encodings"
# ---------------------------

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

class DatasetCapture:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        # Intentar cargar modelos DNN, usar Haar Cascades como fallback
        self.use_dnn = False
        try:
            if (Path('opencv_face_detector_uint8.pb').exists() and 
                Path('opencv_face_detector.pbtxt').exists()):
                self.face_net = cv2.dnn.readNetFromTensorflow('opencv_face_detector_uint8.pb', 
                                                             'opencv_face_detector.pbtxt')
                self.use_dnn = True
                print("Usando detección DNN (más precisa)")
            else:
                print("Usando Haar Cascades (estándar)")
        except Exception as e:
            print(f"Error cargando DNN, usando Haar Cascades: {str(e)}")
            self.use_dnn = False
        
    def detect_faces_dnn(self, frame):
        """Detecta rostros usando DNN (más preciso que Haar Cascades)"""
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123])
        self.face_net.setInput(blob)
        detections = self.face_net.forward()
        
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:  # umbral de confianza
                x1 = int(detections[0, 0, i, 3] * w)
                y1 = int(detections[0, 0, i, 4] * h)
                x2 = int(detections[0, 0, i, 5] * w)
                y2 = int(detections[0, 0, i, 6] * h)
                
                # Convertir a formato (x, y, w, h)
                x, y, w, h = x1, y1, x2-x1, y2-y1
                
                # Filtrar rostros muy pequeños
                if w >= FACE_SIZE_MIN[0] and h >= FACE_SIZE_MIN[1]:
                    faces.append((x, y, w, h, confidence))
        
        return faces
    
    def detect_faces_haar(self, gray):
        """Detecta rostros usando Haar Cascades (backup)"""
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=FACE_SIZE_MIN,
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Agregar confianza dummy
        faces_with_conf = [(x, y, w, h, 0.8) for (x, y, w, h) in faces]
        return faces_with_conf
    
    def calculate_blur_score(self, image):
        """Calcula el nivel de desenfoque de una imagen"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()
    
    def extract_face_with_context(self, frame, face_coords):
        """Extrae el rostro con contexto adicional"""
        x, y, w, h, confidence = face_coords
        
        # Calcular márgenes adaptativos (30% del tamaño del rostro)
        margin_w = int(w * 0.3)
        margin_h = int(h * 0.3)
        
        # Expandir la región de interés
        top = max(0, y - margin_h)
        left = max(0, x - margin_w)
        bottom = min(frame.shape[0], y + h + margin_h)
        right = min(frame.shape[1], x + w + margin_w)
        
        face_region = frame[top:bottom, left:right]
        
        return face_region, {
            'confidence': confidence,
            'blur_score': self.calculate_blur_score(face_region),
            'size': (w, h),
            'position': (x, y)
        }
    
    def extract_face_encoding(self, face_image):
        """Extrae características faciales (encoding) usando el mismo método del entrenamiento"""
        try:
            # Redimensionar a tamaño estándar
            face_resized = cv2.resize(face_image, (64, 64))
            
            # Convertir a escala de grises si es color
            if len(face_resized.shape) == 3:
                gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
            else:
                gray = face_resized
            
            # Asegurar que sea uint8
            gray = gray.astype(np.uint8)
            
            # Normalizar la imagen
            gray = cv2.equalizeHist(gray)
            
            # Aplicar suavizado
            gray = cv2.GaussianBlur(gray, (3, 3), 0)
            
            # Extraer características simples - dividir en regiones y calcular estadísticas
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
            
            # Completar hasta 128 características
            while len(caracteristicas) < 128:
                caracteristicas.append(0.0)
            
            # Tomar solo las primeras 128
            caracteristicas = caracteristicas[:128]
            
            # Normalizar
            caracteristicas = np.array(caracteristicas, dtype=np.float32)
            if np.linalg.norm(caracteristicas) > 0:
                caracteristicas = caracteristicas / np.linalg.norm(caracteristicas)
            
            return caracteristicas
            
        except Exception as e:
            print(f"Error extrayendo características: {e}")
            return None
    
    def save_encodings_to_mongodb(self, person_name, encodings_list):
        """Guarda las huellas faciales en MongoDB"""
        if collection is None:
            print("No hay conexión a MongoDB")
            return False
        
        try:
            # Verificar si ya existe la persona
            existing_person = collection.find_one({'person_name': person_name})
            
            if existing_person:
                print(f"La persona '{person_name}' ya existe en MongoDB")
                response = input("¿Quieres sobrescribir los datos existentes? (s/n): ")
                if response.lower() != 's':
                    print("Operación cancelada")
                    return False
                
                # Eliminar datos existentes
                collection.delete_one({'person_name': person_name})
                print(f"Datos anteriores de '{person_name}' eliminados")
            
            # Preparar documento para MongoDB
            document = {
                'person_name': person_name,
                'face_encodings': [encoding.tolist() for encoding in encodings_list],
                'timestamp': time.time(),
                'capture_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'total_encodings': len(encodings_list),
                'encoding_dimension': 128,
                'extraction_method': 'opencv_custom_features',
                'quality_scores': []  # Aquí se pueden agregar scores de calidad si se desea
            }
            
            # Insertar en MongoDB
            result = collection.insert_one(document)
            print(f"Huellas faciales guardadas en MongoDB:")
            print(f"   • ID del documento: {result.inserted_id}")
            print(f"   • Persona: {person_name}")
            print(f"   • Total de huellas: {len(encodings_list)}")
            print(f"   • Dimensión de cada huella: 128 características")
            
            return True
            
        except Exception as e:
            print(f"Error guardando en MongoDB: {str(e)}")
            return False
    
    def capture_and_extract_encodings(self, person_name):
        """Función principal: captura imágenes y extrae huellas faciales"""
        print("=" * 60)
        print("CAPTURA Y EXTRACCIÓN DE HUELLAS FACIALES")
        print("Guardar en MongoDB para entrenamiento posterior")
        print("=" * 60)
        
        # Crear carpeta del dataset local (backup)
        person_dir = Path(DATASET_DIR) / person_name
        person_dir.mkdir(parents=True, exist_ok=True)
        print(f"Backup local en: {person_dir}")

        # Inicializar cámara con mejores parámetros
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("ERROR: No se pudo abrir la cámara")
            return False
        
        # Configurar cámara para mejor calidad
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print(f"\nInstrucciones para una captura óptima:")
        print("- Mantén iluminación uniforme")
        print("- Evita sombras fuertes en el rostro")
        print("- Mueve la cabeza lentamente (izq/der, arriba/abajo)")
        print("- Cambia expresiones faciales gradualmente")
        print("- Mantén distancia de 0.5-1.5 metros de la cámara")
        print("- Evita movimientos bruscos")
        
        # Countdown
        print("\nIniciando captura en...")
        for i in range(3, 0, -1):
            print(f"{i}...")
            time.sleep(1)
        print("¡GRABANDO!")
        
        frames_data = []
        face_encodings = []  # Lista para almacenar las huellas faciales
        tiempo_inicio = time.time()
        frame_count = 0
        
        # Flags para mostrar tipo de detección usado
        print(f"Tipo de detección: {'DNN (Avanzada)' if self.use_dnn else 'Haar Cascades (Estándar)'}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            tiempo_transcurrido = time.time() - tiempo_inicio
            if tiempo_transcurrido >= DURACION_VIDEO:
                break
            
            # Detectar rostros
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if self.use_dnn:
                faces = self.detect_faces_dnn(frame)
            else:
                faces = self.detect_faces_haar(gray)
            
            # Procesar el mejor rostro (mayor confianza)
            if faces:
                best_face = max(faces, key=lambda f: f[4])  # Por confianza
                face_region, metadata = self.extract_face_with_context(frame, best_face)
                
                # Solo guardar si cumple criterios de calidad
                if (metadata['blur_score'] > QUALITY_THRESHOLD and 
                    metadata['size'][0] >= FACE_SIZE_MIN[0] and 
                    metadata['size'][1] >= FACE_SIZE_MIN[1]):
                    
                    # Guardar frame para backup local
                    frames_data.append({
                        'frame': face_region.copy(),
                        'timestamp': tiempo_transcurrido,
                        'metadata': metadata
                    })
                    
                    # Extraer huella facial (encoding)
                    encoding = self.extract_face_encoding(face_region)
                    if encoding is not None:
                        face_encodings.append(encoding)
            
            # Visualización mejorada
            self.draw_interface(frame, faces, tiempo_transcurrido, len(face_encodings))
            
            cv2.imshow("Captura Huellas Faciales - Presiona Q para cancelar", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nCaptura cancelada por el usuario")
                break
            
            frame_count += 1
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Guardar backup local
        local_success = self.save_captured_data(frames_data, person_dir)
        
        # Guardar huellas faciales en MongoDB
        if face_encodings:
            mongo_success = self.save_encodings_to_mongodb(person_name, face_encodings)
        else:
            print("No se extrajeron huellas faciales válidas")
            mongo_success = False
        
        # Resumen final
        print(f"\nRESUMEN DE CAPTURA:")
        print(f"   • Frames procesados: {frame_count}")
        print(f"   • Huellas faciales extraídas: {len(face_encodings)}")
        print(f"   • Backup local: {'OK' if local_success else 'ERROR'}")
        print(f"   • Guardado en MongoDB: {'OK' if mongo_success else 'ERROR'}")
        
        if mongo_success:
            print(f"\n¡Proceso completado exitosamente!")
            print(f"   Las huellas faciales están listas para entrenar el modelo")
            print(f"\nSiguiente paso:")
            print(f"   python 02_entrenar_modelo_mongodb.py")
        
        return mongo_success
    
    def draw_interface(self, frame, faces, tiempo_transcurrido, total_encodings):
        """Dibuja la interfaz de usuario durante la captura"""
        h, w = frame.shape[:2]
        
        # Información superior
        tiempo_restante = DURACION_VIDEO - tiempo_transcurrido
        cv2.putText(frame, f"Tiempo: {tiempo_restante:.1f}s", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Huellas extraidas: {total_encodings}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Dibujar rostros detectados
        for x, y, face_w, face_h, confidence in faces:
            color = (0, 255, 0) if confidence > 0.7 else (0, 165, 255)
            cv2.rectangle(frame, (x, y), (x + face_w, y + face_h), color, 2)
            cv2.putText(frame, f"{confidence:.2f}", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Indicador de calidad
        if not faces:
            cv2.putText(frame, "NO SE DETECTA ROSTRO", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Barra de progreso
        progress_width = w - 40
        progress_height = 15
        progress_x = 20
        progress_y = h - 30
        
        cv2.rectangle(frame, (progress_x, progress_y), 
                     (progress_x + progress_width, progress_y + progress_height), 
                     (50, 50, 50), -1)
        
        progress_fill = int((tiempo_transcurrido / DURACION_VIDEO) * progress_width)
        cv2.rectangle(frame, (progress_x, progress_y), 
                     (progress_x + progress_fill, progress_y + progress_height), 
                     (0, 255, 0), -1)
        
        # Indicadores de zona óptima (círculo central)
        center_x, center_y = w // 2, h // 2
        cv2.circle(frame, (center_x, center_y), 150, (255, 255, 255), 2)
        cv2.putText(frame, "Zona optima", (center_x - 60, center_y + 170),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def save_captured_data(self, frames_data, person_dir):
        """Guarda backup local de los frames capturados"""
        if not frames_data:
            print("No se capturaron frames para backup local")
            return False
        
        print(f"\nGuardando backup local de {len(frames_data)} frames...")
        
        # Seleccionar frames distribuidos uniformemente
        num_to_save = min(NUM_FOTOS, len(frames_data))
        step = len(frames_data) / num_to_save
        selected_indices = [int(i * step) for i in range(num_to_save)]
        
        saved_count = 0
        for i, idx in enumerate(selected_indices):
            frame_data = frames_data[idx]
            
            # Formato: persona_YYYYMMDD_HHMMSS_001.jpg
            timestamp_str = time.strftime('%Y%m%d_%H%M%S')
            filename = f"{person_dir.name}_{timestamp_str}_{i:03d}.jpg"
            filepath = person_dir / filename
            
            # Guardar imagen
            success = cv2.imwrite(str(filepath), frame_data['frame'])
            if success:
                saved_count += 1
        
        print(f"   Backup local guardado: {saved_count} imágenes")
        return True

def list_persons_in_database():
    """Lista las personas ya almacenadas en MongoDB"""
    if collection is None:
        print("No hay conexión a MongoDB")
        return
    
    try:
        persons = collection.find({}, {'person_name': 1, 'total_encodings': 1, 'capture_date': 1})
        persons_list = list(persons)
        
        if not persons_list:
            print("No hay personas registradas en la base de datos")
            return
        
        print("\nPERSONAS EN LA BASE DE DATOS:")
        print("-" * 50)
        for i, person in enumerate(persons_list, 1):
            print(f"{i}. {person['person_name']}")
            print(f"   • Huellas: {person.get('total_encodings', 'N/A')}")
            print(f"   • Fecha: {person.get('capture_date', 'N/A')}")
            print()
        
    except Exception as e:
        print(f"Error consultando base de datos: {str(e)}")

def main():
    print("=" * 60)
    print("SISTEMA DE CAPTURA Y ALMACENAMIENTO EN MONGODB")
    print("Paso 1: Captura de huellas faciales")
    print("=" * 60)
    
    # Mostrar personas existentes
    list_persons_in_database()
    
    capture = DatasetCapture()
    
    nombre = input("\nEscribe el nombre de la persona: ").strip()
    if not nombre:
        print("Nombre vacío, saliendo...")
        return
    
    success = capture.capture_and_extract_encodings(nombre)
    
    if success:
        print(f"\n¡Huellas faciales de '{nombre}' guardadas exitosamente!")
        print(f"\nPara ver el estado actual de la base de datos ejecuta:")
        print(f"   python 01_capturar_huellas_mongodb.py")
        print(f"\nPara entrenar el modelo con todos los datos:")
        print(f"   python 02_entrenar_modelo_mongodb.py")

if __name__ == "__main__":
    main()