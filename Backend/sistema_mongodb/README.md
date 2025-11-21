# 🎯 **SISTEMA DE RECONOCIMIENTO FACIAL CON MONGODB**

## 📋 **DESCRIPCIÓN**

Sistema profesional de reconocimiento facial que utiliza MongoDB como base de datos para almacenar huellas faciales y gestionar todo el flujo de entrenamiento y reconocimiento.

### ✨ **CARACTERÍSTICAS PRINCIPALES:**
- 🎥 Captura profesional con detección avanzada de rostros
- 🧠 Extracción de huellas faciales (face encodings) 
- 🏗️ Almacenamiento estructurado en MongoDB
- 📊 Entrenamiento de modelo neural personalizado
- 🎯 Reconocimiento en tiempo real
- 📈 Logs y estadísticas completas

---

## 🔄 **FLUJO DEL SISTEMA**

```
1️⃣ CAPTURA → 2️⃣ ENTRENAMIENTO → 3️⃣ RECONOCIMIENTO
   (MongoDB)      (MongoDB)        (Tiempo Real)
```

### **1️⃣ Captura y Almacenamiento**
- Captura video de 40 segundos por persona
- Extrae huellas faciales de alta calidad
- Almacena en MongoDB para entrenamiento posterior

### **2️⃣ Entrenamiento del Modelo**
- Lee todas las huellas desde MongoDB
- Entrena red neuronal personalizada
- Guarda modelo entrenado (.h5) y clases (.npy)

### **3️⃣ Reconocimiento en Tiempo Real**
- Usa modelo entrenado para predicciones
- Compara con base de datos MongoDB
- Registra logs de reconocimiento

---

## 📁 **ESTRUCTURA DE ARCHIVOS**

```
sistema_mongodb/
├── 01_capturar_huellas_mongodb.py      # Captura y almacenamiento
├── 02_entrenar_modelo_mongodb.py       # Entrenamiento del modelo
├── 03_reconocimiento_tiempo_real_mongodb.py  # Reconocimiento
├── 99_utilidades_mongodb.py            # Gestión de base de datos
├── README.md                           # Este archivo
│
├── modelo_rostros_mongodb.h5           # Modelo entrenado (generado)
├── clases_mongodb.npy                  # Clases del modelo (generado)
└── dataset_raw/                        # Backup local de imágenes
    └── [PersonaX]/
        ├── *.jpg                       # Imágenes de respaldo
        └── capture_metadata.txt        # Metadatos
```

---

## 🚀 **GUÍA DE USO**

### **📋 REQUISITOS PREVIOS:**
```bash
# Activar entorno virtual
.\venv310\Scripts\Activate.ps1

# Verificar dependencias instaladas:
# - opencv-python
# - numpy
# - tensorflow
# - pymongo
# - scikit-learn
```

### **🔥 FLUJO COMPLETO:**

#### **Paso 1: Capturar Primera Persona**
```bash
python 01_capturar_huellas_mongodb.py
# Ingresa el nombre: "Juan Perez"
# Sigue las instrucciones de captura (40 segundos)
```

#### **Paso 2: Capturar Más Personas**
```bash
python 01_capturar_huellas_mongodb.py
# Ingresa el nombre: "Maria Lopez"
# Repite para cada persona (mínimo 2 para entrenar)
```

#### **Paso 3: Entrenar el Modelo**
```bash
python 02_entrenar_modelo_mongodb.py
# Se entrena automáticamente con todos los datos de MongoDB
```

#### **Paso 4: Reconocimiento en Tiempo Real**
```bash
python 03_reconocimiento_tiempo_real_mongodb.py
# Presiona 'Q' para salir, 'S' para estadísticas
```

#### **Paso 5: Gestión y Utilidades**
```bash
python 99_utilidades_mongodb.py
# Menú interactivo para gestionar la base de datos
```

---

## 🗄️ **ESTRUCTURA DE MONGODB**

### **Base de Datos:** `face_recognition_system`

#### **Colección: `face_encodings`**
```json
{
  "_id": "ObjectId",
  "person_name": "Juan Perez",
  "face_encodings": [
    [0.123, 0.456, ...],  // 128 características por huella
    [0.789, 0.012, ...]   // Múltiples huellas por persona
  ],
  "timestamp": 1700000000,
  "capture_date": "2025-11-19 10:30:00",
  "total_encodings": 85,
  "encoding_dimension": 128,
  "extraction_method": "opencv_custom_features"
}
```

#### **Colección: `model_training_history`**
```json
{
  "_id": "ObjectId",
  "training_date": "2025-11-19 11:00:00",
  "model_path": "modelo_rostros_mongodb.h5",
  "training_stats": {
    "total_samples": 450,
    "num_classes": 5,
    "classes": ["Juan Perez", "Maria Lopez", "..."],
    "final_accuracy": 0.96,
    "final_val_accuracy": 0.94,
    "training_time_seconds": 45.2
  }
}
```

#### **Colección: `recognition_logs`**
```json
{
  "_id": "ObjectId",
  "person_detected": "Juan Perez",
  "confidence": 0.89,
  "timestamp": 1700000000,
  "recognition_date": "2025-11-19 11:30:00",
  "model_used": "modelo_rostros_mongodb.h5"
}
```

---

## 🛠️ **UTILIDADES DISPONIBLES**

### **99_utilidades_mongodb.py** incluye:

1. **👥 Listar personas** - Ver todas las personas registradas
2. **🧠 Historial de entrenamientos** - Ver entrenamientos pasados
3. **🎯 Logs de reconocimiento** - Ver reconocimientos recientes  
4. **📊 Estadísticas generales** - Resumen del sistema
5. **🗑️ Eliminar persona** - Remover persona de la BD
6. **🧹 Limpiar logs** - Eliminar logs antiguos
7. **📤 Exportar datos** - Backup en archivos JSON

---

## 📊 **VENTAJAS DE ESTE SISTEMA**

### **✅ Vs Sistema Local:**
- 🌐 **Escalabilidad** - Base de datos centralizada
- 🔄 **Sincronización** - Múltiples dispositivos pueden usar la misma BD
- 📈 **Analytics** - Logs y estadísticas automáticas
- 🔒 **Backup** - Datos seguros en la nube
- 🚀 **Performance** - Consultas optimizadas

### **✅ Vs Bibliotecas Preentrenadas:**
- 🎯 **Personalización** - Modelo entrenado específicamente para tus rostros
- ⚡ **Velocidad** - Modelo ligero y rápido
- 🔧 **Control** - Puedes modificar cualquier aspecto
- 📚 **Aprendizaje** - Entiendes exactamente cómo funciona

---

## 🎯 **MÉTRICAS DE CALIDAD**

### **Durante la Captura:**
- ✅ Detección de rostros con confianza > 50%
- ✅ Análisis de desenfoque automático
- ✅ Tamaño mínimo de rostro: 80x80px
- ✅ Márgenes adaptativos del 30%

### **Durante el Entrenamiento:**
- ✅ Validación cruzada del 10%
- ✅ Early stopping para evitar overfitting
- ✅ Reducción de learning rate adaptativa
- ✅ Normalización de características

### **Durante el Reconocimiento:**
- 🟢 **Confianza > 80%:** Verde - "Nombre"
- 🟡 **Confianza 50-80%:** Amarillo - "Posible Nombre?"  
- 🔴 **Confianza < 50%:** Rojo - "Desconocido"

---

## 🚨 **SOLUCIÓN DE PROBLEMAS**

### **❌ Error de Conexión a MongoDB:**
```
# Verificar URL de conexión en cada script
# Verificar que MongoDB Atlas esté activo
# Verificar conexión a internet
```

### **❌ No se detectan rostros:**
```
# Verificar iluminación adecuada
# Mantener distancia de 0.5-1.5 metros
# Verificar que la cámara funcione
```

### **❌ Baja precisión del modelo:**
```
# Capturar más imágenes por persona (mínimo 50-100)
# Verificar calidad de las imágenes capturadas
# Asegurar variedad en expresiones y ángulos
# Re-entrenar el modelo con más datos
```

### **❌ Archivos del modelo no encontrados:**
```
# Ejecutar: python 02_entrenar_modelo_mongodb.py
# Verificar que se generaron los archivos .h5 y .npy
```

---

## 📈 **EXTENSIONES FUTURAS**

### **🔮 Posibles Mejoras:**
- 🌐 **API REST** - Exponer funcionalidad via web
- 📱 **App móvil** - Cliente para dispositivos móviles
- 🎭 **Detección de emociones** - Análisis de expresiones
- 👥 **Múltiples rostros** - Reconocimiento simultáneo
- 🔐 **Autenticación** - Sistema de acceso seguro
- 📊 **Dashboard web** - Panel de control visual
- 🎥 **Video analytics** - Análisis de videos grabados

---

## 🎉 **¡SISTEMA LISTO!**

¡Has implementado un sistema profesional completo de reconocimiento facial con MongoDB! 

🚀 **Comienza tu primera captura:**
```bash
python 01_capturar_huellas_mongodb.py
```

📧 **Para soporte:** Consulta los logs en MongoDB o usa las utilidades incluidas.

---

*Sistema desarrollado con código personalizado, sin dependencias de bibliotecas de reconocimiento facial preentrenadas.*