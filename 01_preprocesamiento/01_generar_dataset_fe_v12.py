"""
Paso 01: Generación de competencia_01_fe_v12_NA.csv a partir del dataset original de la cátedra.
Construye clase_ternaria (si se parte del crudo sin target), define clase_peligro y preserva NAs.
"""
import os
import duckdb
import numpy as np
import pandas as pd

BASE_PATH = '/content/drive/MyDrive/DMEyF/2026/notebooks/data/'
INPUT_CSV = os.path.join(BASE_PATH, 'competencia_01.csv')
OUTPUT_CSV = os.path.join(BASE_PATH, 'competencia_01_fe_v12_NA.csv')

def main():
    print("Cargando dataset original con DuckDB...")
    con = duckdb.connect()
    con.execute("PRAGMA memory_limit='45GB'")
    df = con.execute(f"SELECT * FROM read_csv_auto('{INPUT_CSV}')").df()
    con.close()

    # Si el archivo es el crudo sin clase_ternaria, se calcula según contigüidad m+1 y m+2
    if 'clase_ternaria' not in df.columns:
        meses = sorted(df['foto_mes'].unique())
        mi = df['foto_mes'].map({m: i for i, m in enumerate(meses)}).tolist()
        cli = df['numero_de_cliente'].tolist()
        presentes = set(zip(cli, mi))
        en_m1 = np.fromiter(((c, m + 1) in presentes for c, m in zip(cli, mi)), bool, len(df))
        en_m2 = np.fromiter(((c, m + 2) in presentes for c, m in zip(cli, mi)), bool, len(df))
        clase = np.where(~en_m1, 'BAJA+1', np.where(~en_m2, 'BAJA+2', 'CONTINUA')).astype(object)
        mi_arr = np.array(mi)
        clase[mi_arr == len(meses) - 1] = None
        clase[(mi_arr == len(meses) - 2) & en_m1] = None
        df['clase_ternaria'] = clase

    # Target binario unificado: BAJA+1 y BAJA+2 como positivos (1)
    df['clase_peligro'] = df['clase_ternaria'].isin(['BAJA+1', 'BAJA+2']).astype(np.int8)

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Dataset exportado exitosamente en: {OUTPUT_CSV} ({df.shape[0]:,} filas, {df.shape[1]} columnas)")

if __name__ == '__main__':
    main()