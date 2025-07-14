# TFG DogFinder V2

Repositorio del Trabajo de Fin de Grado: Identificación de imágenes de perros mediante extracción de características con MobileNetV2

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

## Ejecución rápida y sencilla en Google Colab 🚀

Este proyecto está preparado para que cualquier usuario pueda ejecutarlo fácilmente en Google Colab, sin necesidad de conocimientos avanzados sobre rutas ni gestión manual de archivos.

### ¿Cómo funciona?

1. **Abre el notebook desde el badge "Abrir en Colab" del README.**
2. **Ejecuta la primera celda del notebook, que automáticamente:**
   - Monta tu Google Drive.
   - Descarga y descomprime los datos solo si no existen.
   - Verifica que las carpetas de datos estén disponibles.
3. **¡Listo!** Ya puedes ejecutar el resto del notebook sin preocuparte por subir manualmente las carpetas de imágenes ni configurar rutas.

### Ejemplo de celda inicial automatizada

```python
# =============================================
# INSTRUCCIONES AUTOMÁTICAS PARA EL USUARIO
# Esta celda:
# - Monta Google Drive
# - Descarga y descomprime los datos si no existen
# - Verifica que las carpetas estén disponibles
# =============================================

from google.colab import drive
import os

# Montar Google Drive
drive.mount('/content/drive')

# Ruta donde deben estar los datos
DATA_ROOT = '/content/drive/MyDrive/Dog_Mx_Dataset'

# Si no existen los datos, descarga y descomprime automáticamente
if not os.path.exists(DATA_ROOT):
    print("No se encontraron los datos. Descargando y descomprimiendo...")
    !pip install -q gdown
    !gdown --id 11azv5Jxpnz5AjXQjCIeRFaIkHTJxiNRu -O /content/DogFinderData.zip
    !unzip /content/DogFinderData.zip -d /content/drive/MyDrive/
else:
    print("¡Datos encontrados! No es necesario descargar ni descomprimir.")

# Verifica que la carpeta principal existe
if os.path.exists(DATA_ROOT):
    print("✅ Carpeta de datos disponible:", DATA_ROOT)
    print("Ejemplo de contenido:", os.listdir(DATA_ROOT)[:5])
else:
    print("❌ ERROR: No se encontró la carpeta de datos. Por favor, revisa las instrucciones.")
```

### Explicación de rutas
- **Google Colab accede a tu Google Drive en la ruta `/content/drive/MyDrive/`**.
- Por ejemplo, si tienes una carpeta llamada `Dog_Mx_Dataset` en tu Drive, la ruta completa será `/content/drive/MyDrive/Dog_Mx_Dataset/`.
- Si cambias el nombre o la ubicación de la carpeta raíz, actualiza la variable `DATA_ROOT` en la celda anterior.

### Consejos y advertencias
- El archivo de datos es grande. Asegúrate de tener suficiente espacio en tu Google Drive.
- Si ya tienes las carpetas de datos en tu Drive, la celda lo detectará y no descargará nada extra.
- Si tienes problemas de espacio o red, revisa tu Drive y vuelve a intentar la descarga.
- Si tienes dudas, revisa los mensajes que aparecen al ejecutar la celda: te indicarán si todo está correcto o si falta algún dato.

## Organización recomendada de datos en Google Drive

Para ejecutar el notebook en Google Colab, sube tus carpetas de imágenes a tu Google Drive siguiendo esta estructura:

```
/MyDrive/DogFinderData/
├── Dog_Mx_Dataset/
│   ├── perro1.jpg
│   └── ...
├── Stanford_images/
│   ├── stanford1.jpg
│   └── ...
├── NEW_IMAGES_DIR_RF4/
│   └── ...
├── test_images_limitations/
│   └── ...
├── duplicate_test_subset/
│   └── ...
├── nonDogs/
│   └── ...
```

En el notebook, accede a las imágenes usando rutas como:
```python
from google.colab import drive
drive.mount('/content/drive')

# Ejemplo de ruta a imágenes
path = '/content/drive/MyDrive/DogFinderData/Dog_Mx_Dataset/'
```

## Notas
- El modelo MobileNetV2 se usa como caja negra (no se modifica su código interno).
- El proyecto prioriza la integración eficiente y la optimización de recursos.
- Para pruebas rápidas, usa el dataset reducido (`TEST_IMAGE_DIRS`).

## Créditos
Autor: Daniela Díaz. | TFG INGENIERÍA INFORMÁTICA - UNIR 2025

---
