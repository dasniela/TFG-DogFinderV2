from common_dog_finder_config import DogRecognitionModel
from memory_profiler import profile
import os


@profile
def medir_inferencia_memoria():
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    print("Inicializando modelo...")
    model = DogRecognitionModel()

    img_path = 'nonDogs/cat.jpeg'  

    print("Preprocesando imagen...")
    img = model.preprocess_image(img_path)

    if img is not None:
        print("Extrayendo características...")
        features = model.extract_features(img)
        print("Inferencia completada.")
    else:
        print("ERROR: No se pudo procesar la imagen.")

if __name__ == '__main__':
    medir_inferencia_memoria()