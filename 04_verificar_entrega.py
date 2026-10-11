"""
Paso 04: Verifica que el archivo generado por 03_entrenar_y_predecir_M4_final.py
sea 100% idéntico (SHA256 y clientes) al archivo subido al bot (103.7M).
"""
import os, hashlib, re
import pandas as pd

base_path = '/content/drive/MyDrive/DMEyF/2026/notebooks/data/'
carpeta_m4 = os.path.join(base_path, 'evaluacion_recreacion_final', 'Candidato_B_I11_4M_Base')

ruta_submit_bot = os.path.join(carpeta_m4, 'RECREACION_Candidato_B_I11_4M_Base_K11500.csv')
ruta_reproducida = os.path.join(carpeta_m4, 'ENTREGA_FINAL_M4_4M_Base_K11500.csv')

c_bot = open(ruta_submit_bot, 'rb').read()
c_rep = open(ruta_reproducida, 'rb').read()

sha_bot = hashlib.sha256(c_bot).hexdigest()
sha_rep = hashlib.sha256(c_rep).hexdigest()

ids_bot = set(pd.read_csv(ruta_submit_bot, header=None)[0])
ids_rep = set(pd.read_csv(ruta_reproducida, header=None)[0])
en_comun = len(ids_bot.intersection(ids_rep))

print(f"✔ Clientes en común: {en_comun:,} / 11,500 ({en_comun/115:.2f}%)")
print(f"✔ SHA256 Submit Bot:  {sha_bot}")
print(f"✔ SHA256 Reproducido: {sha_rep}")
assert sha_bot == sha_rep, "Los hashes no coinciden"
print("\n✅ VERIFICACIÓN EXITOSA: Reproducibilidad 100.00% exacta bit a bit.")