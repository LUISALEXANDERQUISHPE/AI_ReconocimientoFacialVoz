# 📊 **FLUJO COMPLETO DE SCRIPTS - SISTEMA PROFESIONAL**

## 🔄 **DIAGRAMA DE FLUJO GENERAL**

```
📸 CAPTURA → 🔧 PROCESAMIENTO → 🧠 ENTRENAMIENTO → 🎯 RECONOCIMIENTO
     │              │                │                 │
     ▼              ▼                ▼                 ▼
   Crudo         Normalizado      Modelo H5        Tiempo Real
```

---

## 📋 **FLUJO DETALLADO POR ETAPAS**

### **ETAPA 1: 📸 CAPTURA DE DATASET**

#### **Opción A: Captura Original (USADO ANTERIORMENTE)**
```bash
python 03_capturar_dataset.py
```
**Flujo interno:**
1. Solicita nombre de persona
2. Graba video 40 segundos con cámara
3. Detecta rostros con Haar Cascades básico
4. Extrae 100 frames distribuidos uniformemente
5. Recorta rostros con márgenes básicos
6. Guarda imágenes directamente en `dataset/`

#### **Opción B: Captura Profesional (NUEVA)**
```bash
python 03_capturar_dataset_profesional.py
```
**Flujo interno mejorado:**
1. Solicita nombre de persona
2. Configura cámara en alta resolución (1280x720, 30fps)
3. Graba video con detección avanzada (Haar + soporte DNN)
4. Filtra frames por calidad (desenfoque, iluminación)
5. Aplica detección con umbrales de confianza
6. Extrae rostros con márgenes adaptativos (30%)
7. Guarda en `dataset_raw/` con metadatos completos

---

### **ETAPA 2: 🔧 PROCESAMIENTO DE IMÁGENES**

```bash
python 03b_procesar_imagenes.py --all
# o
python 03b_procesar_imagenes.py "Nombre_Persona"
```

**Flujo de procesamiento:**
```
dataset_raw/[Persona]/*.jpg
         │
         ▼
    🔍 Análisis de Calidad
    ├── Medición de desenfoque (Laplaciano)
    ├── Análisis de brillo/contraste  
    ├── Detección de sobre/sub exposición
    ├── Validación de detección facial
    └── Score global (0-100)
         │
         ▼
    ✨ Mejora de Imagen
    ├── CLAHE (equalización adaptativa)
    ├── Filtro bilateral (reducción ruido)
    ├── Ajuste de contraste/brillo
    └── Normalización facial a 160x160px
         │
         ▼
    🎲 Data Augmentation (si calidad >60)
    ├── Imagen original mejorada
    ├── Rotación ligera (-10° a +10°)
    ├── Ajuste de brillo (±20%)
    └── Flip horizontal
         │
         ▼
    💾 Guardado en dataset/[Persona]/
    ├── [nombre]_processed.jpg (original mejorada)
    ├── [nombre]_rotated.jpg
    ├── [nombre]_brightness.jpg
    ├── [nombre]_flipped.jpg
    └── processing_report.json (metadatos)
```

---

### **ETAPA 3: 🧠 ENTRENAMIENTO DEL MODELO**

```bash
python 04_entrenar_modelo.py
```

**Flujo de entrenamiento:**
```
dataset/[Todas_Personas]/*.jpg
         │
         ▼
    📂 Carga de Imágenes
    ├── Escanea todas las carpetas de personas
    ├── Filtra extensiones válidas (.jpg, .png, etc.)
    └── Carga imágenes con OpenCV
         │
         ▼
    👤 Detección de Rostros
    ├── Aplica Haar Cascades
    ├── Selecciona rostro más grande por imagen
    └── Valida tamaño mínimo (30x30px)
         │
         ▼
    🔢 Extracción de Características
    ├── Redimensiona rostro a 64x64px
    ├── Convierte a escala de grises
    ├── Aplica ecualización de histograma
    ├── Suaviza con filtro Gaussiano
    ├── Divide en regiones 8x8 (64 regiones)
    ├── Calcula estadísticas por región (media, desv.std)
    ├── Agrega estadísticas globales
    └── Normaliza vector a 128 características
         │
         ▼
    🏷️ Preparación de Etiquetas
    ├── Codifica nombres a números (LabelEncoder)
    ├── Convierte a categorical (one-hot encoding)
    └── Valida mínimo 2 clases
         │
         ▼
    🧠 Construcción del Modelo (Keras)
    ├── Input: 128 características
    ├── Dense(256) + BatchNorm + Dropout(0.3)
    ├── Dense(128) + BatchNorm + Dropout(0.3)  
    ├── Dense(64) + Dropout(0.2)
    └── Dense(num_clases) + Softmax
         │
         ▼
    🎯 Entrenamiento
    ├── Optimizador: Adam (lr=0.001)
    ├── Loss: Categorical Crossentropy
    ├── Epochs: Adaptativo (min 20, max 1000)
    ├── Batch Size: Adaptativo (4-32)
    ├── Validación: 10% de datos
    ├── EarlyStopping (patience=10)
    └── ReduceLROnPlateau (patience=5)
         │
         ▼
    💾 Guardado
    ├── modelo_rostros.h5 (arquitectura + pesos)
    └── clases.npy (mapeo nombre-índice)
```

---

### **ETAPA 4: 🎯 RECONOCIMIENTO EN TIEMPO REAL**

```bash
python 05_reconocimiento_tiempo_real.py
```

**Flujo de reconocimiento:**
```
🎥 Cámara en Vivo
         │
         ▼
    📹 Captura de Frame
    ├── Lee frame de cámara
    └── Convierte a escala de grises
         │
         ▼
    👤 Detección de Rostros
    ├── Aplica Haar Cascades
    ├── Filtro: minSize=(60,60)
    └── Para cada rostro detectado:
         │
         ▼
    🔢 Extracción de Características
    ├── MISMA función que entrenamiento
    ├── Redimensiona rostro a 64x64px
    ├── Procesa con pipeline idéntico
    └── Genera vector de 128 características
         │
         ▼
    🧠 Predicción
    ├── Carga modelo_rostros.h5
    ├── Carga clases.npy
    ├── Ejecuta model.predict()
    └── Obtiene probabilidades por clase
         │
         ▼
    🎯 Interpretación de Resultados
    ├── Confianza > 80%: Verde "Persona" 
    ├── Confianza 50-80%: Amarillo "Posible Persona?"
    ├── Confianza < 50%: Rojo "Desconocido"
    └── Dibuja rectángulo + texto en video
         │
         ▼
    📺 Visualización
    ├── Muestra frame con anotaciones
    ├── Loop continuo hasta 'Q'
    └── Libera recursos al salir
```

---

## 🔍 **SCRIPTS DE VERIFICACIÓN Y UTILIDADES**

### **Verificación Visual del Dataset**
```bash
python verificar_dataset.py reporte        # Reporte general
python verificar_dataset.py muestra        # Muestra visual de todas las personas  
python verificar_dataset.py comparar "Edison Fiallos" 5  # Antes vs después
```

### **Migración de Dataset Existente**
```bash
python migrar_dataset.py
```
**Flujo:**
1. Lee dataset existente en `dataset/`
2. Copia a `dataset_raw/` con nomenclatura nueva
3. Respalda original en `dataset_backup_YYYYMMDD_HHMMSS/`
4. Genera `migration_report.json`

### **Descarga de Modelos DNN (Opcional)**
```bash
python descargar_modelos_dnn.py
```

---

## ⚡ **FLUJO COMPLETO RECOMENDADO**

### **Para NUEVO dataset:**
```bash
# 1. Capturar persona por persona
python 03_capturar_dataset_profesional.py  

# 2. Procesar todas las imágenes
python 03b_procesar_imagenes.py --all

# 3. Verificar calidad
python verificar_dataset.py reporte

# 4. Entrenar modelo
python 04_entrenar_modelo.py

# 5. Probar en tiempo real
python 05_reconocimiento_tiempo_real.py
```

### **Para dataset EXISTENTE (YA EJECUTADO):**
```bash
# ✅ 1. Migración (YA HECHO)
python migrar_dataset.py  

# ✅ 2. Procesamiento (YA HECHO)
python 03b_procesar_imagenes.py --all

# 3. Entrenamiento con datos mejorados
python 04_entrenar_modelo.py

# 4. Reconocimiento mejorado
python 05_reconocimiento_tiempo_real.py
```

---

## 📊 **ARCHIVOS GENERADOS EN EL FLUJO**

```
Backend/
├── dataset_raw/                    # 📸 Imágenes crudas de captura
│   └── [Persona]/
│       ├── *.jpg
│       └── capture_metadata.txt
│
├── dataset/                        # 🔧 Imágenes procesadas listas
│   └── [Persona]/  
│       ├── *_processed.jpg         # Imagen principal normalizada
│       ├── *_rotated.jpg           # Augmentación rotación  
│       ├── *_brightness.jpg        # Augmentación brillo
│       ├── *_flipped.jpg           # Augmentación flip
│       └── processing_report.json  # Metadatos de calidad
│
├── modelo_rostros.h5               # 🧠 Modelo entrenado
├── clases.npy                      # 🏷️ Mapeo de nombres
├── migration_report.json           # 📋 Reporte de migración
└── dataset_backup_*/               # 💾 Backup del dataset original
```

---

## 🎯 **TU ESTADO ACTUAL**

✅ **Completado:**
- Migración del dataset existente  
- Procesamiento profesional de 2,400 imágenes
- Calidad promedio: 87.0/100
- Dataset listo para entrenamiento

🔄 **Siguiente paso:**
```bash
python 04_entrenar_modelo.py
```

¡El flujo está optimizado para máxima calidad y profesionalismo! 🚀