import os
import random
import numpy as np
import tensorflow as tf


# Global

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "bbc")

CLASSES = ["business", "entertainment", "politics", "sport", "tech"]
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASSES)}

# Vectorización
MAX_TOKENS = 20000
SEQ_LEN = 300

# Entrenamiento
BATCH_SIZE = 32
EPOCHS = 6
LR = 1e-3

# Funcion   que comprueba que exista el directorio y las subcarpetas
def check_dataset_structure(data_dir):
    if not os.path.isdir(data_dir):
        print(f"El directorio '{data_dir}' no existe.")
        return False

    for cls in CLASSES:
        cls_path = os.path.join(data_dir, cls)
        if not os.path.isdir(cls_path):
            print(f"Falta la carpeta requerida: {cls_path}")
            return False

    print(" Estructura del dataset correcta.")
    return True

