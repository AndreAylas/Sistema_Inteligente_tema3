import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, log_loss, classification_report

# Directorio del csb train300min.csv
CSV_PATH= os.path.join(os.path.dirname(os.path.abspath(__file__)), 'train300mil.csv')

def read_data(n_rows=10_000):
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(
            f"No se encontró el fichero: {CSV_PATH}\n"
        )
    # Leemos solo n_rows
    df = pd.read_csv(CSV_PATH, nrows=n_rows)
    print(f"[INFO] Dataset cargado: filas={len(df)} columnas={len(df.columns)}")
    return df

def split_xy(df):
    if "click" not in df.columns:
        raise KeyError("No existe la columna 'click' (target) en el CSV.")

    X = df.drop(columns=["click"])
    y = df["click"]
    return X, y


def build_preprocessor(X):
    # Categóricas = tipo object. Numéricas = resto
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

    print(f"[INFO] Columnas categóricas: {len(cat_cols)} | numéricas: {len(num_cols)}")

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ],
        remainder="drop"
    )
    return preprocessor

def train_and_evaluate(model_name, clf, preprocessor, X_train, X_test, y_train, y_test):
    pipe= Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', clf)
    ])# instanciamos el pipeline

    print(f" entrenando el modelo: {model_name}...")

    pipe.fit(X_train, y_train)
    y_pred=pipe.predict(X_test)
    y_prob=None
    if hasattr(pipe.named_steps["classifier"],"predict_proba"):
        y_prob=pipe.predict_proba(X_test)
    acc=accuracy_score(y_test,y_pred) # exactitud del modelo
    print(f"resultado del modelo: {acc:.4f}")

    if y_prob is not None:
        ll = log_loss(y_test, y_prob)
        print(f"[RESULT] Log loss: {ll:.4f}")
    else:
        print(" el clasficador no exponwe predict_proba; no se calcula log_loss. ")
    
    print("\n Clasificacion Report:")
    print(classification_report(y_test,y_pred, digits=4))

    return pipe

def main():
    # 1.- Leer 10.000 filas del csv
    df= read_data(n_rows=10_000) # leer n filas del csv
    X, y= split_xy(df) # separar en X e y
    preprocessor=build_preprocessor(X) # preprocesador

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f" entrenamiento: {X_train.shape}, test: {X_test.shape}")

    # modelo -> sin regularización
    clf_no_reg= SGDClassifier(
        loss="log_loss",
        penalty=None,
        max_iter=10,
        eta0=0.01,
        learning_rate="constant",
        random_state=42
    )
    pipe1=train_and_evaluate(
        model_name="SGD (sin regularización)",
        clf=clf_no_reg,
        preprocessor=preprocessor, 
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test
    )

    # modelo --> regularzacion L1
    clf_l1= SGDClassifier(
        loss="log_loss",
        penalty="l1",
        alpha=0.0001,
        max_iter=10,
        eta0=0.01,
        learning_rate="constant",
        random_state=42
    )

    pipe2= train_and_evaluate(
        model_name="SGD (L1, alpha=0.0001)",
        clf=clf_l1,
        preprocessor=preprocessor,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test
    )

if __name__ == "__main__":
    main()