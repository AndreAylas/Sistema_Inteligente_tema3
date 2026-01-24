import pandas as pd
import os

CSV_PATH= os.path.join(os.path.dirname(__file__),"compact.csv")

def read_csv():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(
            "No se encontro compact.csv. descargalo segun el enunciado del ejercicio"
        )
    return pd.read_csv(CSV_PATH)
    
def get_country(df,country_name="spain"):
    """
        funcion que lee la country he imprime todas las columnas
    """
    if "country" not in df.columns:
        raise KeyError("No existe  la columna country en el csv...")
    mask=df["country"].astype(str).str.strip().str.lower() ==country_name.strip().lower()
    df_country =df[mask].copy()
    
    if df_country.empty:
        raise ValueError(f"No se encontro filas para {country_name} en la columna country ..")
    
    print(f"Pais filtrado: {country_name} tiene numero de filas: {len(df_country)} y columnas: {len(df_country.columns)}")
    return df_country

def clean_country(df_country):
    """
        b) Limpia el DataFrame del país (España) 
    """
    df_clean=df_country.copy()
    
    # ii) date -> datetime
    if "date" not in df_clean.columns:
        raise KeyError("No existe la columna 'date' en el DataFrame.")
    df_clean["date"] = pd.to_datetime(df_clean["date"], errors="coerce")

    # i) NaN -> 0 solo en columnas numéricas
    num_cols = df_clean.select_dtypes(include="number").columns
    df_clean[num_cols] = df_clean[num_cols].fillna(0)

    # iii) filas con new_cases negativo
    if "new_cases" not in df_clean.columns:
        raise KeyError("No existe la columna 'new_cases' en el DataFrame.")
    negativos = df_clean[df_clean["new_cases"] < 0].copy()

    print("\n Filas con new_cases < 0 ")
    if negativos.empty:
        print("No hay filas con new_cases negativo.")
    else:
        # Mostrar columnas clave si existen
        cols_mostrar = [c for c in ["country", "date", "new_cases", "total_cases"] if c in negativos.columns]
        print(negativos[cols_mostrar].to_string(index=False))
        print(f"Total filas con new_cases negativo: {len(negativos)}")

    # iv) duplicados
    dup_antes = df_clean.duplicated().sum()
    print(f"\nDuplicados antes: {dup_antes}")

    df_clean = df_clean.drop_duplicates().copy()

    dup_despues = df_clean.duplicated().sum()
    print(f"Duplicados después: {dup_despues}")

    return df_clean, negativos
    
def save_data_frame(df, filename="covid_espana.xlsx"):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(base_dir, filename)

    df.to_excel(output_path, index=False)
    print(f"\n[INFO] Excel guardado en: {output_path}")

    # Comprobación
    df_check = pd.read_excel(output_path)
    print("[INFO] Comprobación: leído de vuelta el Excel. Primeras 5 filas:")


    return output_path
if __name__ == '__main__':
    df=read_csv()
    
    print("\n[OK] csv leido correctamente...")
    # apartado a)
    df_spain=get_country(df,"spain")
    #aparatado b)
    df_clean,negativo=clean_country(df_spain)
    save_data_frame(df_clean)
    