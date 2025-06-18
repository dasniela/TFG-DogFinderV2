# TFG DogFinder V2

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dasniela/TFG-DogFinderV2/blob/main/TFG_DogFinder_FeatureExtraction%20.ipynb)


Repositorio del Trabajo de Fin de Grado: Búsqueda de similitud de imágenes de perros mediante extracción de características con MobileNetV2

## Objetivo
Sistema de reconocimiento de imágenes y búsqueda de similitud diseñado para identificar y categorizar perros basándose en sus características visuales. Utiliza un modelo de aprendizaje profundo pre-entrenado (MobileNetV2) para extraer características de las imágenes y la librería FAISS para realizar búsquedas eficientes de similitud en una base de datos de perros registrados.


### Características principales:
- Uso de MobileNetV2 como extractor de características (sin modificar su código)
- Scripts para agregar imágenes y procesarlas en la base de datos
- Integración con base de datos SQLite y FAISS para búsquedas rápidas
- Organización de imágenes por grupos (ver `dog_groups.json`)
- Configuración centralizada en `common_dog_finder_config.py`

### Funciones principales
- Extracción de Características: Utiliza el modelo MobileNetV2 pre-entrenado para generar vectores numéricos (embeddings) que representan las características visuales únicas de cada perro.
- Gestión de Base de Datos (SQLite): Almacena información detallada de los perros (nombre, ubicación, fecha, ruta de imagen y sus características extraídas) en una base de datos local.
- Detección de Duplicados: Al añadir imágenes, el sistema verifica automáticamente si la imagen ya existe (por ruta o por similitud de características) para evitar registros redundantes.
- Búsqueda de Similitud Visual (FAISS): Permite encontrar rápidamente los perros más similares a una imagen de consulta dentro de la base de datos, utilizando un índice FAISS optimizado para búsquedas a gran escala.
- Evaluación de Rendimiento: Incluye herramientas para evaluar la precisión del sistema de búsqueda en diferentes umbrales de similitud, utilizando métricas como Precision, Recall y F1-Score.

## Estructura del Proyecto
- `add_dogs_to_db.py`: Script principal para procesar y registrar imágenes en la base de datos, usando FAISS para evitar duplicados.
- `common_dog_finder_config.py`: Configuración global, modelos, rutas y utilidades para el reconocimiento y la base de datos.
- `dog_features.faiss`: Índice FAISS para búsqueda eficiente de imágenes por similitud.
- `dog_finder_demo_v4.db`: Base de datos SQLite con los registros de perros procesados.
- `dog_id_map.json`: Mapeo entre IDs de FAISS y rutas de imágenes.
- `dog_groups.json`: Grupos de imágenes de perros con caracteristicas similares para el "ground truth" de las pruebas de evaluación.
- `Dog_Mx_Dataset/`, `Stanford_images/`, `Tsinghua_Dogs_Dataset/`: Carpetas con datasets de imágenes de perros.
- `TFG_DogFinder_FeatureExtraction .ipynb`: Notebook para experimentación y extracción de características.

## Uso rápido
1. Instala las dependencias necesarias (TensorFlow, SQLAlchemy, FAISS, OpenCV, etc.)
2. Ajusta los directorios de imágenes y rutas en `common_dog_finder_config.py` si es necesario.
3. Ejecuta `add_dogs_to_db.py` para poblar la base de datos y el índice FAISS.
4. Consulta los scripts y notebooks para búsqueda y pruebas.

## Notas
- El modelo MobileNetV2 se usa como caja negra (no se modifica su código interno).
- El proyecto prioriza la integración eficiente y la optimización de recursos.
- Para pruebas rápidas, usa el dataset reducido (`TEST_IMAGE_DIRS`).

## Créditos
Autor: Daniela Díaz. | TFG 2025

---
