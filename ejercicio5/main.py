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

# leer texto
def read_text_file(path):
    try:
        with open(path,'r',encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        with open(path, "r", encoding="latin-1", errors="ignore") as f:
            return f.read()

def load_texts_and_labels(data_dir):
    texts=[]
    labels=[]
    
    # leer las subcarpetas del directorio bbc
    for _class in CLASSES:
        class_dir=os.path.join(data_dir,_class)
        files=[f for f in os.listdir(class_dir) if f.lower().endswith(".txt")]
        files.sort()
        
        for fname in files:
            fpath=os.path.join(class_dir,fname)
            txt = read_text_file(fpath).strip()
            if not txt:
                continue
            texts.append(txt)
            labels.append(CLASS_TO_ID[_class])
        
    if len(texts)==0:
        print("[ERROR] No se cargo ningun texto revisar si existe texto en las subcarpetas")
        return [], np.array([],dtype=np.int32)
    
    print(f"[INFO] total de documentos guardados: {len(texts)}")
    
    counts={c:0 for c in CLASSES}
    for y in labels:
        counts[CLASSES[y]]+=1
    print(f"[INFO] Distribucion por clases: {counts}")
    
    return texts, np.array(labels, dtype=np.int32)

def split_data(texts,labels,train_ratio=0.8,val_ratio=0.1):
    idx=list(range(len(texts))) # genera una lista de los indices de texts
    random.shuffle(idx) # desordena la lista
    
    n= len(idx)
    n_train=int(n*train_ratio)
    n_val=int(n*val_ratio)
    
    train_idx=idx[:n_train]
    val_idx=idx[n_train:n_train+n_val]
    test_idx=idx[n_train+n_val:]
    
    X_train=[texts[i] for i in train_idx]
    y_train=labels[train_idx]
    
    X_val = [texts[i] for i in val_idx]
    y_val = labels[val_idx]

    X_test = [texts[i] for i in test_idx]
    y_test = labels[test_idx]

    print(f"[INFO] Split -> train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")
    return X_train, y_train, X_val, y_val, X_test, y_test

def make_tf_dataset(texts,labels, batch_size, shuffle=False):
    ds= tf.date.Dataset.from_tensor_slice((texts,labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(texts), 2000), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

def build_vectorizer(max_tokens=MAX_TOKENS,seq_len=SEQ_LEN):
    vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=max_tokens,
        output_mode="int",
        output_sequence_length=seq_len,
        standardize="lower_and_strip_punctuation",
        split="whitespace"
    )
    return vectorizer

def build_model(vectorizer):
    inputs = tf.keras.Input(shape=(1,), dtype=tf.string, name="text")
    x = vectorizer(inputs)
    x = tf.keras.layers.Embedding(input_dim=MAX_TOKENS, output_dim=128, name="emb")(x)
    x = tf.keras.layers.Conv1D(filters=128, kernel_size=5, activation="relu", name="conv")(x)
    x = tf.keras.layers.GlobalMaxPooling1D(name="gmp")(x)
    x = tf.keras.layers.Dense(128, activation="relu", name="dense1")(x)
    x = tf.keras.layers.Dropout(0.4, name="drop")(x)
    outputs = tf.keras.layers.Dense(len(CLASSES), activation="softmax", name="out")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="bbc_text_cnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )
    return model

def train_model(model, train_ds, val_ds, epochs=EPOCHS):
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)
    ]
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )

def evaluate_model(model, test_ds):
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    print(f"\n[RESULT] Test loss: {test_loss:.4f}")
    print(f"[RESULT] Test accuracy: {test_acc:.4f}")
    return test_loss, test_acc


def predict_example(model, text):
    probs = model.predict([text], verbose=0)[0]
    pred_id = int(np.argmax(probs))
    return pred_id, probs

def main():
    if not check_dataset_structure(DATA_DIR):
        print("[INFO] Proceso terminado: estructura del dataset incorrecto")
        return
    texts, labels= load_texts_and_labels(DATA_DIR)
    