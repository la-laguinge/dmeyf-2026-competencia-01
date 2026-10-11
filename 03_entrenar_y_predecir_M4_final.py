"""
Paso 03: Entrenamiento y Predicción del Modelo Final M4 (4 Meses Base + Ensamble 3 Semillas).
Apunta directamente a la carpeta donde se guardaron los modelos .txt de la recreación ganadora
para garantizar reproducción 100% exacta bit a bit (o re-entrenar si USAR_MODELOS_GUARDADOS=False).
"""
import os, gc, duckdb, datetime, time
import pandas as pd, numpy as np
import lightgbm as lgb
from lightgbm import LGBMClassifier
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 0. RUTAS Y CONFIGURACIÓN
# ==========================================
base_path = '/content/drive/MyDrive/DMEyF/2026/notebooks/data/'
carpeta_m4 = os.path.join(base_path, 'evaluacion_recreacion_final', 'Candidato_B_I11_4M_Base')
modelos_dir = os.path.join(carpeta_m4, 'modelos')
os.makedirs(modelos_dir, exist_ok=True)

dataset_path = os.path.join(base_path, 'competencia_01_fe_v12_NA.csv')

# True: Carga los 3 árboles .txt congelados en modelos/ (Reproducción 100.00% idéntica SHA256)
# False: Re-entrena las 3 semillas desde cero en GPU
USAR_MODELOS_GUARDADOS = True

cols_basura = ['clase_ternaria', 'numero_de_cliente', 'foto_mes', 'peso_temporal']
SEMILLAS = [578777, 377717, 717797]
K_EVAL = 11500

top_9_vars = [
    'ctrx_quarter', 'mcaja_ahorro', 'mtarjeta_visa_consumo', 'mcuentas_saldo',
    'mpayroll', 'cpayroll_trx', 'mprestamos_personales', 'mactivos_margen', 
    'Visa_mlimitecompra'
]

PARAMS_M4 = {
    'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt', 'is_unbalance': True,
    'n_jobs': -1, 'verbose': -1, 'device': 'gpu', 
    'max_bin': 31, 'min_data_in_leaf': 0, 'learning_rate': 0.005, 
    'n_estimators': 2098, 'num_leaves': 33, 'min_sum_hessian_in_leaf': 0.0763,
    'feature_fraction': 0.5831, 'feature_fraction_bynode': 0.2806,
    'bagging_fraction': 0.7040, 'bagging_freq': 5
}

# ==========================================
# 1. CARGA E INGENIERÍA DE ATRIBUTOS (TOP 9 LAGS & DELTAS)
# ==========================================
print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] 📂 Cargando datos e ingeniería M4 (4 Meses Base)...")
con = duckdb.connect()
con.execute("PRAGMA memory_limit='45GB'")
df = con.execute(f"SELECT * FROM read_csv_auto('{dataset_path}') WHERE foto_mes IN (202101, 202102, 202103, 202104, 202105, 202106, 202108)").df()
con.close()

for c in df.columns:
    if df[c].dtype == 'float64': df[c] = df[c].astype('float32', copy=False)
if 'ccajas_depositos' in df.columns:
    df.loc[df['foto_mes'] == 202105, 'ccajas_depositos'] = np.nan

df.sort_values(by=['numero_de_cliente', 'foto_mes'], inplace=True)
df.reset_index(drop=True, inplace=True)

cols_originales = [c for c in df.columns if c not in cols_basura + ['clase_peligro']]
es_mismo_1 = (df['numero_de_cliente'] == df['numero_de_cliente'].shift(1)).values
es_mismo_2 = (df['numero_de_cliente'] == df['numero_de_cliente'].shift(2)).values

base_top9 = []
for col in top_9_vars:
    if col in df.columns:
        v = df[col].values
        l1 = np.where(es_mismo_1, df[col].shift(1).values, np.nan).astype(np.float32)
        l2 = np.where(es_mismo_2, df[col].shift(2).values, np.nan).astype(np.float32)
        df[f'{col}_lag1'] = l1; df[f'{col}_lag2'] = l2
        df[f'{col}_delta1'] = np.where(es_mismo_1, v - l1, np.nan).astype(np.float32)
        df[f'{col}_delta2'] = np.where(es_mismo_2, v - l2, np.nan).astype(np.float32)
        base_top9.extend([f'{col}_lag1', f'{col}_lag2', f'{col}_delta1', f'{col}_delta2'])

cols_activas = cols_originales + base_top9

mask_tr = df['foto_mes'].isin([202103, 202104, 202105, 202106])
mask_te = df['foto_mes'] == 202108

X_tr = df.loc[mask_tr, cols_activas]
y_tr = df.loc[mask_tr, 'clase_peligro'].fillna(0).astype(np.int8).values
X_te = df.loc[mask_te, cols_activas]
num_cliente = df.loc[mask_te, 'numero_de_cliente'].values

del df; gc.collect()

# ==========================================
# 2. PREDICCIÓN DESDE MODELOS GUARDADOS O RE-ENTRENAMIENTO
# ==========================================
probas = np.zeros(len(X_te), dtype=np.float64)

for seed in SEMILLAS:
    ruta_txt = os.path.join(modelos_dir, f'lgb_seed_{seed}.txt')
    if USAR_MODELOS_GUARDADOS and os.path.exists(ruta_txt):
        print(f"   📦 Cargando modelo congelado: lgb_seed_{seed}.txt")
        booster = lgb.Booster(model_file=ruta_txt)
        prob = booster.predict(X_te)
    else:
        print(f"   🚀 Entrenando semilla {seed}...")
        p = PARAMS_M4.copy()
        p['random_state'] = seed
        try:
            m = LGBMClassifier(**p)
            m.fit(X_tr, y_tr)
        except Exception:
            p.pop('device', None)
            m = LGBMClassifier(**p)
            m.fit(X_tr, y_tr)
        m.booster_.save_model(ruta_txt)
        prob = m.predict_proba(X_te)[:, 1]
        
    probas += prob / len(SEMILLAS)

df_ens = pd.DataFrame({'numero_de_cliente': num_cliente, 'prob': probas}).sort_values(by='prob', ascending=False, kind='mergesort')
ruta_salida = os.path.join(carpeta_m4, f"ENTREGA_FINAL_M4_4M_Base_K{K_EVAL}.csv")
df_ens.head(K_EVAL)['numero_de_cliente'].astype(int).to_csv(ruta_salida, index=False, header=False)

print(f"\n[{datetime.datetime.now().strftime('%H:%M:%S')}] 🏁 Archivo final exportado en:\n   {ruta_salida}")