import os
import numpy as np
import cv2
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras import layers, models
from pymongo import MongoClient
import time

# ---------------- CONFIGURACIÓN ----------------
MONGO_URI = "mongodb+srv://lithubprogramadores_db_user:NlejQAZ9OuLJnv55@ai.xr4bewc.mongodb.net/?appName=Ai"
DB_NAME = "face_recognition_system_Scripts"
COLLECTION_NAME = "face_encodings"
MODELO_PATH = "modelo_rostros_mongodb.h5"
CLASES_PATH = "clases_mongodb.npy"
# ------------------------------------------------

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

def load_encodings_from_mongodb():
    """Carga las huellas faciales y etiquetas desde MongoDB"""
    if collection is None:
        print("No hay conexión a MongoDB")
        return np.array([]), np.array([])
    
    try:
        print("Cargando datos desde MongoDB...")
        
        # Obtener todos los documentos
        documents = list(collection.find({}))
        
        if not documents:
            print("No se encontraron datos en MongoDB")
            print("   Ejecuta primero: python 01_capturar_huellas_mongodb.py")
            return np.array([]), np.array([])
        
        X = []  # Características (huellas faciales)
        y = []  # Etiquetas (nombres)
        
        print(f"Documentos encontrados en MongoDB: {len(documents)}")
        print("\nProcesando personas:")
        
        for doc in documents:
            person_name = doc['person_name']
            face_encodings = doc['face_encodings']
            
            print(f"• {person_name}: {len(face_encodings)} huellas")
            
            # Convertir cada encoding de lista a numpy array y agregarlo
            for encoding in face_encodings:
                encoding_array = np.array(encoding, dtype=np.float32)
                X.append(encoding_array)
                y.append(person_name)
        
        # Convertir a numpy arrays
        X = np.array(X, dtype=np.float32)
        y = np.array(y)
        
        print(f"\nDATOS CARGADOS EXITOSAMENTE:")
        print(f"• Total de huellas faciales: {len(X)}")
        print(f"• Dimensión de cada huella: {X.shape[1] if len(X) > 0 else 0}")
        
        # Mostrar distribución por persona
        if len(y) > 0:
            unique_people, counts = np.unique(y, return_counts=True)
            print(f"   • Distribución por persona:")
            for person, count in zip(unique_people, counts):
                print(f"     - {person}: {count} huellas")
        
        return X, y
        
    except Exception as e:
        print(f"Error cargando datos desde MongoDB: {str(e)}")
        return np.array([]), np.array([])

def construir_modelo(num_clases):
    """Construir modelo de red neuronal para clasificación de rostros"""
    model = models.Sequential([
        layers.Input(shape=(128,)),
        
        # Capa de entrada con más neuronas
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # Segunda capa
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # Tercera capa
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        
        # Capa de salida
        layers.Dense(num_clases, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

def save_training_results_to_mongodb(training_stats):
    """Guarda estadísticas del entrenamiento en MongoDB"""
    try:
        training_collection = db["model_training_history"]
        
        training_doc = {
            'training_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'timestamp': time.time(),
            'model_path': MODELO_PATH,
            'classes_path': CLASES_PATH,
            'training_stats': training_stats,
            'model_architecture': 'Custom CNN Sequential',
            'optimizer': 'Adam',
            'loss_function': 'categorical_crossentropy'
        }
        
        result = training_collection.insert_one(training_doc)
        print(f"Estadísticas de entrenamiento guardadas en MongoDB: {result.inserted_id}")
        
    except Exception as e:
        print(f"Error guardando estadísticas: {str(e)}")

def main():
    print("=" * 60)
    print("ENTRENAMIENTO DE MODELO DESDE MONGODB")
    print("Paso 2: Entrenar modelo con huellas faciales almacenadas")
    print("=" * 60)
    
    print("Cargando huellas faciales desde MongoDB...")
    X, y = load_encodings_from_mongodb()

    if len(X) == 0:
        print("\nERROR: No se encontraron datos válidos en MongoDB.")
        print("\nPara resolver esto:")
        print("1. Ejecuta: python 01_capturar_huellas_mongodb.py")
        print("2. Captura al menos 2 personas diferentes")
        print("3. Vuelve a ejecutar este script")
        return

    print(f"\nDatos cargados exitosamente desde MongoDB:")
    print(f"  Total de muestras: {X.shape[0]}")
    print(f"  Dimensión de cada huella: {X.shape[1]}")

    # Codificar etiquetas de texto a números
    le = LabelEncoder()
    y_int = le.fit_transform(y)
    clases = le.classes_
    num_clases = len(clases)

    print(f"  Clases encontradas: {list(clases)}")
    print(f"  Número de clases: {num_clases}")
    
    if num_clases < 2:
        print("\nERROR: Se necesitan al menos 2 personas diferentes para entrenar.")
        print("Captura más personas con: python 01_capturar_huellas_mongodb.py")
        return

    y_cat = tf.keras.utils.to_categorical(y_int, num_classes=num_clases)

    # Construir modelo
    print(f"\nConstruyendo modelo de red neuronal...")
    model = construir_modelo(num_clases)
    
    print("\nArquitectura del modelo:")
    model.summary()

    # Entrenar modelo
    print(f"\nIniciando entrenamiento...")
    
    # Ajustar epochs basado en la cantidad de datos
    epochs = min(1000, max(20, len(X) * 2))
    batch_size = max(4, min(32, len(X) // 4))
    
    print(f"  Configuración de entrenamiento:")
    print(f"    • Epochs: {epochs}")
    print(f"    • Batch size: {batch_size}")
    print(f"    • Validación: 10% de los datos")
    
    start_time = time.time()
    
    history = model.fit(
        X, y_cat,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        verbose=1,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5)
        ]
    )
    
    training_time = time.time() - start_time

    # Guardar modelo y clases
    print(f"\nGuardando modelo entrenado...")
    model.save(MODELO_PATH)
    np.save(CLASES_PATH, clases)

    print(f"\nENTRENAMIENTO COMPLETADO!")
    print(f"  Archivo del modelo: {MODELO_PATH}")
    print(f"  Archivo de clases: {CLASES_PATH}")
    print(f"  Tiempo de entrenamiento: {training_time:.2f} segundos")
    
    # Mostrar métricas finales
    final_acc = history.history['accuracy'][-1]
    print(f"  Precisión final: {final_acc:.2%}")
    
    val_acc = None
    if 'val_accuracy' in history.history:
        val_acc = history.history['val_accuracy'][-1]
        print(f"  Precisión validación: {val_acc:.2%}")
    
    # Preparar estadísticas para MongoDB
    training_stats = {
        'total_samples': len(X),
        'num_classes': num_clases,
        'classes': list(clases),
        'epochs_executed': len(history.history['accuracy']),
        'final_accuracy': float(final_acc),
        'final_val_accuracy': float(val_acc) if val_acc else None,
        'training_time_seconds': training_time,
        'batch_size': batch_size,
        'data_source': 'mongodb',
        'encoding_dimension': X.shape[1]
    }
    
    # Guardar estadísticas en MongoDB
    save_training_results_to_mongodb(training_stats)
    
    # Evaluación de calidad del modelo
    print(f"\nEVALUACIÓN DE CALIDAD DEL MODELO:")
    if final_acc >= 0.95:
        print(f"   EXCELENTE - Precisión muy alta ({final_acc:.1%})")
    elif final_acc >= 0.85:
        print(f"   BUENA - Precisión aceptable ({final_acc:.1%})")
    elif final_acc >= 0.70:
        print(f"   REGULAR - Podría necesitar más datos ({final_acc:.1%})")
    else:
        print(f"   BAJA - Necesita más datos o ajustes ({final_acc:.1%})")
    
    if val_acc and abs(final_acc - val_acc) > 0.15:
        print(f"   POSIBLE OVERFITTING - Gran diferencia entre entrenamiento y validación")
    
    print(f"\nEl modelo está listo para reconocimiento en tiempo real!")
    print(f"\nSiguiente paso:")
    print(f"   python 03_reconocimiento_tiempo_real_mongodb.py")
    
    return True

if __name__ == "__main__":
    main()