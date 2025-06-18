# add_dogs_to_db.py

import os
import sys
import json
from datetime import datetime
import numpy as np
import faiss

# Importa el módulo common_dog_finder_config como 'cfg'
# Esto nos permitirá acceder a sus variables globales actualizadas
import common_dog_finder_config as cfg

# Importa las clases y funciones específicas, pero no los globales faiss_index_global o faiss_id_map_global, accederemos vía cfg.faiss_index_global
from common_dog_finder_config import (
    DogRecognitionModel, Dog, IMAGE_DIRS, TEST_IMAGE_DIRS, session,
    initialize_faiss_index_from_db, build_and_save_faiss_index,
    FAISS_INDEX_PATH, FAISS_ID_MAP_PATH, DUPLICATE_THRESHOLD_SIMILARITY # Importamos el umbral aquí
)

# La función add_images_to_database NO recibirá los índices globales como argumentos.
def add_images_to_database(image_dirs):
    """
    Procesa y registra imágenes de los directorios especificados en la base de datos.
    Utiliza un índice FAISS para optimizar la detección de duplicados por características,
    incluyendo la detección de duplicados dentro del mismo lote de imágenes que se está procesando.
    """
    model = DogRecognitionModel()
    count_new = 0
    count_exist_path = 0
    count_exist_features = 0

    # Inicializa el índice FAISS global (o cargarlo desde disco).
    # Esto poblará las variables GLOBALES cfg.faiss_index_global y cfg.faiss_id_map_global.
    initialize_faiss_index_from_db()

    # --- Inicializacion de  current_batch_faiss_index ---
    feature_dimension = 1280 # Dimensión de las características de MobileNetV2

    current_batch_faiss_index = faiss.IndexFlatIP(feature_dimension)
    
    # Verificamos si el cfg.faiss_index_global (el global verdadero) tiene elementos
    if cfg.faiss_index_global is not None and cfg.faiss_index_global.ntotal > 0:
        # Reconstruimos los elementos del índice global y los añadimos al índice del batch
        existing_features_from_global = cfg.faiss_index_global.reconstruct_n(0, cfg.faiss_index_global.ntotal)
        current_batch_faiss_index.add(existing_features_from_global)
        print(f"DEBUG: Índice de batch inicializado con {current_batch_faiss_index.ntotal} elementos de la DB existente.")
    else:
        print("DEBUG: Índice de batch iniciado vacío, ya que la DB existente está vacía.")

    # --- Mapeo temporal para elementos añadidos en este batch (para depuración) ---
    current_batch_temp_id_map = [] # Guarda los image_path de las imágenes que se añaden al current_batch_faiss_index en este lote.

    # Prepara un set de paths existentes para verificación rápida por ruta.
    existing_paths_set = set(d.image_path for d in session.query(Dog.image_path).distinct().all())

    print("Iniciando procesamiento de imágenes...")

    try:
        for image_dir in image_dirs:
            for root, _, files in os.walk(image_dir):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_path = os.path.join(root, file)

                        # --- Primera verificacion: duplicado por ruta en la DB ---
                        if image_path in existing_paths_set:
                            count_exist_path += 1
                            continue

                        # Procesa la imagen y extrae características
                        features = model.process_image(image_path)
                        if features is None:
                            print(f"Advertencia: No se pudieron extraer características de {image_path}. Omitiendo.")
                            continue

                        # --- Segunda verificacion: duplicado por caracteristicas (FAISS)  ---
                        is_duplicate_in_current_context = False
                        if current_batch_faiss_index.ntotal > 0: # Solo busca si hay elementos en el índice
                            query_feature_2d = features.astype('float32').reshape(1, -1)
                            
                            distances, indices = current_batch_faiss_index.search(query_feature_2d, k=1)
                            
                            if distances[0][0] >= DUPLICATE_THRESHOLD_SIMILARITY:
                                similar_item_info = ""
                                faiss_idx = indices[0][0] # Índice FAISS del elemento más cercano

                                # Accede a faiss_id_map_global a través de 'cfg.'
                                if faiss_idx < len(cfg.faiss_id_map_global):
                                    # El duplicado está entre las imágenes que ya estaban en la DB
                                    duplicate_dog_db_id = cfg.faiss_id_map_global[faiss_idx]
                                    similar_dog = session.query(Dog).filter_by(id=duplicate_dog_db_id).first()
                                    if similar_dog:
                                        similar_item_info = f"con '{os.path.basename(similar_dog.image_path)}' (ID DB: {duplicate_dog_db_id})"
                                    else:
                                        similar_item_info = f"con ID de FAISS: {faiss_idx} (ya en BD, pero no se encontró en la sesión actual de la DB)"
                                elif faiss_idx >= len(cfg.faiss_id_map_global) and \
                                     (faiss_idx - len(cfg.faiss_id_map_global)) < len(current_batch_temp_id_map):
                                    # El duplicado está entre las imágenes añadidas en este mismo lote de procesamiento
                                    temp_map_idx = faiss_idx - len(cfg.faiss_id_map_global)
                                    similar_item_info = f"con '{os.path.basename(current_batch_temp_id_map[temp_map_idx])}' (previamente en este lote)"
                                else:
                                    similar_item_info = f"con un elemento FAISS de índice: {faiss_idx} (fuente de mapeo desconocida)"

                                print(f"[OMITIENDO] '{os.path.basename(image_path)}' es un duplicado por características (similitud: {distances[0][0]:.2f}) {similar_item_info}. ")
                                count_exist_features += 1
                                is_duplicate_in_current_context = True
                        
                        if is_duplicate_in_current_context:
                            continue

                        # --- Aquí se crea el objeto Dog con la información completa ---
                        name = os.path.basename(file) 
                        location = "Desconocida" 
                        found_date = datetime.now() # La fecha de hallazgo es la fecha actual
                        features_json = json.dumps(features.tolist())
                        
                        print(f"[OK] Añadiendo: {image_path}")

                        dog = Dog(
                            name=name,
                            location=location,
                            found_date=found_date,
                            image_path=image_path,
                            features=features_json
                        )
                        session.add(dog)
                        count_new += 1

                        # --- Actualizacion de current_batch_faiss_index y su mapeo temporal ---
                        current_batch_faiss_index.add(features.astype('float32').reshape(1, -1))
                        current_batch_temp_id_map.append(image_path) # Añade la ruta de la imagen al mapeo temporal
                                                
        session.commit() # Realiza el commit de todas las adiciones

        if count_new > 0:
            print(f"Se añadieron {count_new} nuevas imágenes. Reconstruyendo índice FAISS global para sincronizar con la DB.")
            # Esta llamada a initialize_faiss_index_from_db() actualizará los verdaderos globales
            initialize_faiss_index_from_db() 
            print("Índice FAISS global actualizado y guardado.")
        else:
            print("No se añadieron nuevas imágenes en este lote. El índice FAISS global no necesita reconstruirse.")

        print(f"\n--- Resumen del procesamiento ---")
        print(f"Nuevas imágenes añadidas a la BD: {count_new}")
        print(f"Imágenes omitidas (ya existen por ruta): {count_exist_path}")
        print(f"Imágenes omitidas (duplicados por características): {count_exist_features}")

    except Exception as e:
        session.rollback()
        print(f"Error al añadir imágenes a la base de datos: {e}")
        # Asegura que el índice global se reinicie si hubo un rollback y los datos en DB no coinciden
        initialize_faiss_index_from_db(force_rebuild=True)
    finally:
        session.close()


if __name__ == "__main__":
    try:
        from common_dog_finder_config import IMAGE_DIRS as current_image_dirs
        
        print("Iniciando ejecución directa de add_dogs_to_db.py...")
        
        add_images_to_database(current_image_dirs)
        
    except ImportError as e:
        print(f"Error de importación: {e}")
        print("Asegúrate de que common_dog_finder_config.py esté en la misma carpeta o en el PYTHONPATH.")
        sys.exit(1)
    except Exception as e:
        print(f"Un error inesperado ocurrió durante la ejecución: {e}")