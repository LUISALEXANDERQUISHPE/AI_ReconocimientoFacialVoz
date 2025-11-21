from pymongo import MongoClient
import numpy as np
import json
from datetime import datetime

# --- CONFIGURACIÓN ---
MONGO_URI = "mongodb+srv://lithubprogramadores_db_user:NlejQAZ9OuLJnv55@ai.xr4bewc.mongodb.net/?appName=Ai"
DB_NAME = "face_recognition_system"
# ---------------------

try:
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    print("Conexión a MongoDB exitosa")
except Exception as e:
    print(f"Error conectando a MongoDB: {str(e)}")
    exit()

def listar_personas():
    """Lista todas las personas en la base de datos"""
    collection = db["face_encodings"]
    
    try:
        persons = list(collection.find({}, {
            'person_name': 1, 
            'total_encodings': 1, 
            'capture_date': 1,
            'timestamp': 1
        }))
        
        if not persons:
            print("No hay personas registradas en la base de datos")
            return
        
        print(f"\nPERSONAS EN LA BASE DE DATOS ({len(persons)}):")
        print("-" * 60)
        for i, person in enumerate(persons, 1):
            print(f"{i}. {person['person_name']}")
            print(f"   • Huellas faciales: {person.get('total_encodings', 'N/A')}")
            print(f"   • Fecha de captura: {person.get('capture_date', 'N/A')}")
            print()
            
    except Exception as e:
        print(f"Error consultando base de datos: {str(e)}")

def ver_historial_entrenamientos():
    """Muestra el historial de entrenamientos del modelo"""
    collection = db["model_training_history"]
    
    try:
        trainings = list(collection.find({}).sort('timestamp', -1).limit(10))
        
        if not trainings:
            print("No hay entrenamientos registrados")
            return
        
        print(f"\nHISTORIAL DE ENTRENAMIENTOS (últimos 10):")
        print("-" * 70)
        for i, training in enumerate(trainings, 1):
            stats = training.get('training_stats', {})
            print(f"{i}. Entrenamiento - {training.get('training_date', 'N/A')}")
            print(f"   • Precisión final: {stats.get('final_accuracy', 0):.2%}")
            print(f"   • Precisión validación: {stats.get('final_val_accuracy', 0):.2%}")
            print(f"   • Total de muestras: {stats.get('total_samples', 0)}")
            print(f"   • Clases: {stats.get('num_classes', 0)} ({', '.join(stats.get('classes', []))})")
            print(f"   • Tiempo de entrenamiento: {stats.get('training_time_seconds', 0):.2f}s")
            print()
            
    except Exception as e:
        print(f"Error consultando entrenamientos: {str(e)}")

def ver_logs_reconocimiento():
    """Muestra los últimos reconocimientos realizados"""
    collection = db["recognition_logs"]
    
    try:
        logs = list(collection.find({}).sort('timestamp', -1).limit(20))
        
        if not logs:
            print("No hay reconocimientos registrados")
            return
        
        print(f"\nÚLTIMOS RECONOCIMIENTOS (20 más recientes):")
        print("-" * 60)
        for i, log in enumerate(logs, 1):
            print(f"{i}. {log.get('recognition_date', 'N/A')}")
            print(f"   • Persona: {log.get('person_detected', 'N/A')}")
            print(f"   • Confianza: {log.get('confidence', 0):.1%}")
            print()
            
    except Exception as e:
        print(f"Error consultando logs: {str(e)}")

def estadisticas_generales():
    """Muestra estadísticas generales del sistema"""
    try:
        # Contar personas
        personas_count = db["face_encodings"].count_documents({})
        
        # Contar entrenamientos
        trainings_count = db["model_training_history"].count_documents({})
        
        # Contar reconocimientos
        recognitions_count = db["recognition_logs"].count_documents({})
        
        # Último entrenamiento
        last_training = db["model_training_history"].find().sort('timestamp', -1).limit(1)
        last_training = list(last_training)
        
        # Reconocimientos por persona (últimos 100)
        recent_logs = list(db["recognition_logs"].find({}).sort('timestamp', -1).limit(100))
        
        print(f"\nESTADÍSTICAS GENERALES DEL SISTEMA:")
        print("-" * 50)
        print(f"Personas registradas: {personas_count}")
        print(f"Entrenamientos realizados: {trainings_count}")
        print(f"Reconocimientos registrados: {recognitions_count}")
        
        if last_training:
            training = last_training[0]
            stats = training.get('training_stats', {})
            print(f"\nÚltimo entrenamiento:")
            print(f"   • Fecha: {training.get('training_date', 'N/A')}")
            print(f"   • Precisión: {stats.get('final_accuracy', 0):.2%}")
            print(f"   • Clases entrenadas: {stats.get('num_classes', 0)}")
        
        if recent_logs:
            # Conteo por persona en los últimos reconocimientos
            person_counts = {}
            for log in recent_logs:
                person = log.get('person_detected', 'Desconocido')
                person_counts[person] = person_counts.get(person, 0) + 1
            
            print(f"\nReconocimientos recientes (últimos 100):")
            for person, count in sorted(person_counts.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / len(recent_logs)) * 100
                print(f"   • {person}: {count} ({percentage:.1f}%)")
                
    except Exception as e:
        print(f"Error calculando estadísticas: {str(e)}")

def eliminar_persona():
    """Elimina una persona de la base de datos"""
    listar_personas()
    
    nombre = input("\nEscribe el nombre exacto de la persona a eliminar: ").strip()
    if not nombre:
        print("Nombre vacío")
        return
    
    # Buscar la persona
    collection = db["face_encodings"]
    person = collection.find_one({'person_name': nombre})
    
    if not person:
        print(f"No se encontró la persona '{nombre}'")
        return
    
    # Confirmar eliminación
    print(f"\nVAS A ELIMINAR:")
    print(f"   • Persona: {nombre}")
    print(f"   • Huellas: {person.get('total_encodings', 0)}")
    print(f"   • Fecha: {person.get('capture_date', 'N/A')}")
    
    confirm = input(f"\n¿Estás seguro? Escribe 'ELIMINAR' para confirmar: ")
    if confirm != 'ELIMINAR':
        print("Operación cancelada")
        return
    
    # Eliminar
    try:
        result = collection.delete_one({'person_name': nombre})
        if result.deleted_count > 0:
            print(f"Persona '{nombre}' eliminada exitosamente")
            print(f"Recuerda entrenar el modelo nuevamente si era importante")
        else:
            print(f"No se pudo eliminar")
    except Exception as e:
        print(f"Error eliminando: {str(e)}")

def limpiar_logs():
    """Limpia los logs de reconocimiento antiguos"""
    collection = db["recognition_logs"]
    
    try:
        count = collection.count_documents({})
        print(f"Logs de reconocimiento actuales: {count}")
        
        if count == 0:
            print("No hay logs para limpiar")
            return
        
        confirm = input(f"\n¿Eliminar TODOS los {count} logs? (s/n): ")
        if confirm.lower() != 's':
            print("Operación cancelada")
            return
        
        result = collection.delete_many({})
        print(f"{result.deleted_count} logs eliminados")
        
    except Exception as e:
        print(f"Error eliminando logs: {str(e)}")

def exportar_datos():
    """Exporta los datos a archivos JSON"""
    try:
        # Exportar personas
        personas = list(db["face_encodings"].find({}))
        for person in personas:
            person['_id'] = str(person['_id'])  # Convertir ObjectId a string
        
        with open('export_personas.json', 'w', encoding='utf-8') as f:
            json.dump(personas, f, indent=2, ensure_ascii=False)
        
        # Exportar entrenamientos
        trainings = list(db["model_training_history"].find({}))
        for training in trainings:
            training['_id'] = str(training['_id'])
        
        with open('export_entrenamientos.json', 'w', encoding='utf-8') as f:
            json.dump(trainings, f, indent=2, ensure_ascii=False)
        
        # Exportar logs recientes
        logs = list(db["recognition_logs"].find({}).sort('timestamp', -1).limit(1000))
        for log in logs:
            log['_id'] = str(log['_id'])
        
        with open('export_reconocimientos.json', 'w', encoding='utf-8') as f:
            json.dump(logs, f, indent=2, ensure_ascii=False)
        
        print(f"Datos exportados exitosamente:")
        print(f"   • export_personas.json - {len(personas)} personas")
        print(f"   • export_entrenamientos.json - {len(trainings)} entrenamientos") 
        print(f"   • export_reconocimientos.json - {len(logs)} reconocimientos")
        
    except Exception as e:
        print(f"Error exportando: {str(e)}")

def main():
    print("=" * 60)
    print("UTILIDADES DEL SISTEMA MONGODB")
    print("Gestión y consulta de la base de datos")
    print("=" * 60)
    
    while True:
        print(f"\nOPCIONES DISPONIBLES:")
        print(f"1. Listar personas registradas")
        print(f"2. Ver historial de entrenamientos")
        print(f"3. Ver logs de reconocimiento")
        print(f"4. Estadísticas generales")
        print(f"5. Eliminar persona")
        print(f"6. Limpiar logs de reconocimiento")
        print(f"7. Exportar datos a JSON")
        print(f"0. Salir")
        
        try:
            opcion = input(f"\nElige una opción (0-7): ").strip()
            
            if opcion == "1":
                listar_personas()
            elif opcion == "2":
                ver_historial_entrenamientos()
            elif opcion == "3":
                ver_logs_reconocimiento()
            elif opcion == "4":
                estadisticas_generales()
            elif opcion == "5":
                eliminar_persona()
            elif opcion == "6":
                limpiar_logs()
            elif opcion == "7":
                exportar_datos()
            elif opcion == "0":
                print("¡Hasta luego!")
                break
            else:
                print("Opción no válida")
                
        except KeyboardInterrupt:
            print("\n¡Hasta luego!")
            break
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()