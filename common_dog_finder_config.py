# common_dog_finder_config.py

import os
import numpy as np
import cv2
import json
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base 
from sqlalchemy.orm import sessionmaker
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt 

# --- Importa FAISS ---
try:
    import faiss
except ImportError:
    print("ADVERTENCIA: FAISS no está instalado.")
    faiss = None 

# --- Configuración General ---
DB_PATH = 'sqlite:///dog_finder_demo_v4.db' #Define la ruta de la base de datos SQLite.
IMAGE_DIRS = [ 'Stanford_images', 'Dog_Mx_Dataset'] # lista de directorios con los datasets
FAISS_INDEX_PATH = 'dog_features.faiss' # Ruta para guardar/cargar el índice FAISS. Este índice almacena vectores de características de imágenes
FAISS_ID_MAP_PATH = 'dog_id_map.json'   # Ruta para guardar/cargar el mapeo de IDs. Ruta a un archivo JSON que guarda el mapeo entre los vectores del índice FAISS y los identificadores de las imágenes

# --- Variable auxiliar para pruebas rápidas con dataset pequeño ---
TEST_IMAGE_DIRS = ['Dog_Mx_Dataset']

# Define un umbral de similitud para considerar dos imágenes como duplicados.
DUPLICATE_THRESHOLD_SIMILARITY = 0.95 

# --- Modelo de Reconocimiento ---

#INICIALIZACIÓN DEL MODELO
class DogRecognitionModel:
    def __init__(self):
        base_model = MobileNetV2(
            input_shape=(224, 224, 3), # Tamaño estándar para MobileNetV2 (alto, ancho, canales)
            include_top=False,         # CONGELAMIENTO DE CAPA DE CLASIFICACIÓN: se desactiva la capa de clasificación
            pooling='avg',             # CREACIÓN DE UN VECTOR DE CARACTERISTICAS: Pooling promedio para obtener vector de características
            weights='imagenet'         # CARGAR PESOS PREENTRENADOS DE IMAGENET
        )
        
        # CONGELAMIENTO DE CAPAS PREENTRENADAS: Se congela los pesos para usar solo como extractor
        base_model.trainable = False

        # Definir el modelo extractor
        self.feature_extractor = Model(inputs=base_model.input, outputs=base_model.output)

    #PREPROCESAMIENTO DE IMAGENES
    def preprocess_image(self, image_path):
        # 1. Cargar imagen con OpenCV
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f'No se pudo cargar la imagen: {image_path}')

        # 2. Convertir de BGR a RGB (OpenCV usa BGR por defecto, modelos entrenados con ImageNet requieren RGB)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 3. Redimensionar a 224x224 (tamaño requerido por MobileNetV2)
        img = cv2.resize(img, (224, 224))

        # 4. Convertir a float32 para procesamiento
        img = img.astype('float32')
        
        # 5. Expandir dimensiones para lote (modelo espera lote aún si solo será una imagen)
        img = np.expand_dims(img, axis=0)

        # 6. Preprocesar para MobileNetV2 (normaliza a [-1,1])
        return preprocess_input(img)
    
    def extract_features(self, image):
        # 1. Extraer características usando el modelo.
        # verbose controla la verbosidad durante la predicción. (el valor 0 significa modo silencioso, sin mostrar ninguna salida de progreso).
        features = self.feature_extractor.predict(image, verbose=0)

        # 2. Normalizar el vector (importante para comparación coseno)
        # Dividimos por la norma para obtener un vector unitario
        features = features / np.linalg.norm(features)

        return features

    def process_image(self, image_path):
        # Método que combina preprocesamiento y extracción
        img = self.preprocess_image(image_path)
        return self.extract_features(img)

# --- Configuración de Base de Datos ---

# Se crea una clase base declarativa, de ella heredan todas las clases que representen tablas en la base de datos
Base = declarative_base()
class Dog(Base):
    __tablename__ = 'dogs'
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=True)
    location = Column(String(500), nullable=False)
    found_date = Column(DateTime, nullable=False)
    image_path = Column(String(1000), nullable=False)
    features = Column(String, nullable=False) # JSON string

# Inicialización del motor y la sesión de SQLAlchemy
engine = create_engine(DB_PATH)  #Se crea un "motor" de SQLAlchemy que funciona como la interfaz  para la DB y maneja la conexión.
Base.metadata.create_all(engine)  # crea la tabla dogs en la base de datos especificada por el engine, si no existen.

# Crea una sesion para interctuar con la DB 
Session = sessionmaker(bind=engine)
session = Session() #Se crea una instancia de la sesión para realizar las operaciones en la DB.

# --- Variables globales para FAISS Index ---
# Declara estas variables como globales para que puedan ser modificadas por funciones como initialize_faiss_index_from_db
global faiss_index_global
global faiss_id_map_global
faiss_index_global = None     # Esta variable va a almacenar el índice FAISS una vez que se cargue desde disco o se construya en memoria.
faiss_id_map_global = []      # Esta lista servirá para mapear cada vector del índice FAISS a un identificador.

# --- Funciones FAISS para construir/cargar el índice ---
def build_and_save_faiss_index(features_list_flat, dog_ids, dimension, index_path, id_map_path):
    """
    Construye un índice FAISS a partir de características y lo guarda en disco.
    También guarda el mapeo de IDs.
    features_list_flat: lista de vectores de caracteristicas (arrays NumPy de tipo float32)
    dog_ids: Lista de los dog.id correspondientes  a cada vector.
    dimension: número de dimensiones de cada vector (debe coincidir con la longitud de los vectores).
    index_path: ruta para guardar el archivo FAISS (.faiss).
    id_map_path: ruta para guardar el mapeo de IDs (.json).
    """
    # Si la lista de características está vacía, se imprime una advertencia y no se construye nada.
    if not features_list_flat:
        print("No hay características para construir el índice FAISS.")
        # Asegurarse de que los archivos de índice no existan si no hay datos
        if os.path.exists(index_path):
            os.remove(index_path)
        if os.path.exists(id_map_path):
            os.remove(id_map_path)
        return None

    features_array = np.array(features_list_flat).astype('float32')
    if features_array.shape[1] != dimension:
        raise ValueError(f"Dimensión de las características ({features_array.shape[1]}) no coincide con la dimensión esperada ({dimension}).")

    # Usamos IndexFlatIP para búsqueda por similitud coseno (producto interno).
    index = faiss.IndexFlatIP(dimension)
    index.add(features_array)

    faiss.write_index(index, index_path)
    with open(id_map_path, 'w') as f:
        json.dump(dog_ids, f)
    print(f"Índice FAISS guardado en {index_path} con {index.ntotal} vectores.")
    print(f"Mapeo de IDs guardado en {id_map_path}.")
    return index

def load_faiss_index(index_path, id_map_path):
    # Carga un índice FAISS y su mapeo de IDs desde disco.
    
    if not os.path.exists(index_path) or not os.path.exists(id_map_path):
        print(f"Archivos de índice FAISS no encontrados: {index_path} o {id_map_path}. Se intentará reconstruir.")
        return None, []

    try:
        index = faiss.read_index(index_path)
        with open(id_map_path, 'r') as f:
            id_map = json.load(f)
        print(f"Índice FAISS cargado desde {index_path} con {index.ntotal} vectores.")
        print(f"Mapeo de IDs cargado desde {id_map_path}.")
        return index, id_map
    except Exception as e:
        print(f"Error al cargar el índice FAISS: {e}. Se intentará reconstruir.")
        return None, []

def initialize_faiss_index_from_db(force_rebuild=False):
    """
    Carga o reconstruye el índice FAISS global a partir de la base de datos.
    Si force_rebuild=True, siempre reconstruye el índice desde la base de datos,
    incluso si ya existe un índice cargado en disco.
    Si force_rebuild=False (por defecto), intenta cargar el índice desde disco y solo reconstruye si no existe o está vacío/desactualizado.
    """
    global faiss_index_global
    global faiss_id_map_global

    if faiss is None:
        print("FAISS no está disponible. El índice FAISS no se inicializará.")
        return

    # Intenta cargar si no se fuerza la reconstrucción y los archivos existen
    if not force_rebuild and os.path.exists(FAISS_INDEX_PATH) and os.path.exists(FAISS_ID_MAP_PATH):
        temp_index, temp_id_map = load_faiss_index(FAISS_INDEX_PATH, FAISS_ID_MAP_PATH)
        if temp_index is not None and temp_index.ntotal > 0:
            db_count = session.query(Dog).count()
            # Si el índice tiene menos elementos que la DB, es obsoleto y necesita reconstruirse
            if temp_index.ntotal < db_count:
                print(f"ADVERTENCIA: La DB tiene {db_count} registros, pero el índice FAISS cargado tiene {temp_index.ntotal}. Reconstruyendo índice para sincronizar.")
                force_rebuild = True # Forzar reconstrucción si está desactualizado
            else:
                faiss_index_global = temp_index
                faiss_id_map_global = temp_id_map
                print("Índice FAISS cargado y sincronizado con la base de datos.")
                return # Salir si se cargó correctamente y está sincronizado

    # Si llegamos aquí, necesitamos reconstruir el índice
    print("Reconstruyendo índice FAISS desde la base de datos...")
    existing_dogs = session.query(Dog).all()
    features_to_add = []
    dog_ids_to_map = []
    feature_dimension = 1280 # Dimensión de MobileNetV2 features

    for dog in existing_dogs:
        try:
            features = np.array(json.loads(dog.features)).flatten()
            if features.shape[0] == feature_dimension: 
                features_to_add.append(features)
                dog_ids_to_map.append(dog.id)
            else:
                print(f"Advertencia: Características del perro ID {dog.id} tienen dimensión incorrecta ({features.shape[0]}). Omitiendo.")
        except json.JSONDecodeError as e:
            print(f"Error decodificando características para el perro ID {dog.id}: {e}. Omitiendo.")
        except Exception as e:
            print(f"Error procesando características para el perro ID {dog.id}: {e}. Omitiendo.")

    if features_to_add:
        # Pasa los datos como un array 2D a build_and_save_faiss_index
        faiss_index_global = build_and_save_faiss_index(features_to_add, dog_ids_to_map, feature_dimension, FAISS_INDEX_PATH, FAISS_ID_MAP_PATH)
        faiss_id_map_global = dog_ids_to_map
    else:
        faiss_index_global = None
        faiss_id_map_global = []
        print("Base de datos vacía o sin características válidas, no se puede construir índice FAISS. Borrando archivos de índice si existen.")
        # Limpiar archivos de índice si no hay datos para construir
        if os.path.exists(FAISS_INDEX_PATH):
            os.remove(FAISS_INDEX_PATH)
        if os.path.exists(FAISS_ID_MAP_PATH):
            os.remove(FAISS_ID_MAP_PATH)

def is_duplicate_image_by_features(new_features, threshold=0.95):
    """
    Verifica si una nueva imagen es duplicada comparando su producto interno
    (similitud coseno) con las imágenes ya en el índice FAISS global.
    threshold: Umbral para el producto interno de 0.95
    """

    if faiss is None or faiss_index_global is None or faiss_index_global.ntotal == 0:
        print("ADVERTENCIA: El índice FAISS global no está inicializado o está vacío para la verificación de duplicados.")
        return False

    # new_features debe ser 2D (1, dimension) y float32
    query_feature_2d = new_features.astype('float32').reshape(1, -1)
    
    # Realiza la búsqueda: k=1 para encontrar el más cercano
    distances, indices = faiss_index_global.search(query_feature_2d, k=1)

    # Si la similitud al vecino más cercano es MAYOR O IGUAL que el umbral, consideramos que es un duplicado
    if distances[0][0] >= threshold:
        return True
    return False


def search_similar_dogs(query_features, k=15, threshold_ip=0.50, exclude_dog_id=None): # Umbral justificado 0.50
    """
    Busca perros similares en la base de datos usando el índice FAISS global.
    Retorna los top K perros más similares cuyo producto interno 
    sea mayor o igual al umbral 0.50.
    Solo se devuelven los resultados que cumplen similarity >= threshold_ip.
    """
    if faiss is None or faiss_index_global is None or faiss_index_global.ntotal == 0:
        print("El índice FAISS no está inicializado o está vacío. No se puede realizar la búsqueda de similitud.")
        return []

    # query_features debe ser 2D (1, dimension) y float32
    query_features_2d = query_features.astype('float32').reshape(1, -1)

    k_search_initial = min(max(k*2, k), faiss_index_global.ntotal) # Buscar más para filtrar por umbral

    # Realiza la búsqueda FAISS (producto interno)
    similarities, indices = faiss_index_global.search(query_features_2d, k=k_search_initial)
    similarities = similarities[0]
    indices = indices[0]

    # Filtra por umbral de similitud
    results = []
    for sim, idx in zip(similarities, indices):
        if sim >= threshold_ip:
            if idx == -1: # FAISS devuelve -1 si no encuentra suficientes vecinos.
                continue
            
            # Se asegura de que el índice esté dentro del rango de faiss_id_map_global
            if idx < len(faiss_id_map_global):
                dog_id = faiss_id_map_global[idx]
                if exclude_dog_id is None or dog_id != exclude_dog_id:
                    results.append({"dog_id": dog_id, "similarity": sim})
                    if len(results) >= k:
                        break
            else:
                # Esto no debería pasar si faiss_id_map_global está bien sincronizado
                print(f"ADVERTENCIA: Índice FAISS {idx} fuera de rango para faiss_id_map_global (len={len(faiss_id_map_global)}).")
    return results


def show_image(image_path, title=None):
    #Función para mostrar una imagen usando matplotlib y OpenCV.
 
    img = cv2.imread(image_path)
    if img is None:
        print(f'No se pudo cargar la imagen: {image_path}')
        return

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(8, 6 ))
    plt.imshow(img)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()
    
def reset_database():
    #Borra todos los registros de la tabla 'dogs' 
    print("Borrando todos los registros de la tabla 'dogs'...")
    session.query(Dog).delete()
    session.commit()
    print("Registros borrados.")

    # Borra también los archivos del índice FAISS para empezar de nuevo
    if os.path.exists(FAISS_INDEX_PATH):
        os.remove(FAISS_INDEX_PATH)
        print(f"Archivo de índice FAISS '{FAISS_INDEX_PATH}' borrado.")
    if os.path.exists(FAISS_ID_MAP_PATH):
        os.remove(FAISS_ID_MAP_PATH)
        print(f"Archivo de mapeo de IDs '{FAISS_ID_MAP_PATH}' borrado.")

    # Resetea el índice global en memoria 
    global faiss_index_global
    global faiss_id_map_global
    faiss_index_global = None
    faiss_id_map_global = []

    print("Base de datos e índice FAISS reiniciados.")