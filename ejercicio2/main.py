import pandas as pd
import os

CSV_PATH= os.path.join(os.path.dirname(__file__),"compact.csv")

def read_csv():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(
            "No se encontro compact.csv. descargalo segun el enunciado del ejercicio"
        )
    return pd.read_csv(CSV_PATH)
    
def probando(df):#funciona
    cols= df.columns.tolist()
    
    if "country" in cols and "date" in cols:
        print("Mostrando country y date (primeras 10 filas):")
        print(df[["country", "date"]].head(10).to_string(index=False))
    else:
        print("No existen columnas 'country' y/o 'date'. Mostrando las 2 primeras columnas del CSV:")
        print(df.iloc[:, :2].head(10).to_string(index=False))
def get_country(df,country_name="spain"):
    """
        funcion que lee la country he imprime todas las columnas
    """
    if "country" not in df.columns:
        raise KeyError("No existe  la columna country en el csv...")
    mask=df["country"].astype(str).str.strip().str.lower() ==country_name.strip().lower()
    df_country =df[mask].copy()
    
    if df_country.empty:
        raise ValueError("No se encontro filas para {country_name} en la columna country ..")
    
    print(f"Pais filtrado: {country_name} tiene numero de filas: {len(df_country)} y columnas: {len(df_country.columns)}")
    return df_country
    
if __name__ == '__main__':
    df=read_csv()
    
    print("\n[OK] csv leido correctamente...")
    #probando(df)
    df_spain=get_country(df,"spain")