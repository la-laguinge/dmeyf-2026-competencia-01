"""
Paso 02: Optimización Bayesiana (Origen de los hiperparámetros LowTop2 del modelo M4).
Entrena en 202103-202105 y valida ganancia directa en 202106 (corte K=11500).
"""
import os, gc, duckdb, datetime
import pandas as pd, numpy as np
import optuna
from lightgbm import LGBMClassifier
import warnings

warnings.filterwarnings('ignore')

base_path = '/content/drive/MyDrive/DMEyF/2026/notebooks/data/'
run_base_path = os.path.join(base_path, 'run_v64_Optuna_Bimodal_20min')
os.makedirs(run_base_path, exist_ok=True)

dataset_path = os.path.join(base_path, 'competencia_01_fe_v12_NA.csv')
cols_basura = ['clase_ternaria', 'numero_de_cliente', 'foto_mes', 'peso_temporal']

top_vars = [
    'ctrx_quarter', 'mcaja_ahorro', 'mtarjeta_visa_consumo', 'mcuentas_saldo',
    'mpayroll', 'cpayroll_trx', 'mprestamos_personales', 'mactivos_margen',
    'Visa_mlimitecompra'
]

print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Preparando datos para Optuna (202103-202106)...")
con = duckdb.connect()
con.execute("PRAGMA memory_limit='45GB'")
df = con.execute(f"SELECT * FROM read_csv_auto('{dataset_path}')").df()
con.close()

for c in df.columns:
    if df[c].dtype == 'float64':
        df[c] = df[c].astype('float32', copy=False)

if 'ccajas_depositos' in df.columns:
    df.loc[df['foto_mes'] == 202105, 'ccajas_depositos'] = np.nan

df.sort_values(by=['numero_de_cliente', 'foto_mes'], inplace=True)
df.reset_index(drop=True, inplace=True)

es_mismo_cliente_1 = (df['numero_de_cliente'] == df['numero_de_cliente'].shift(1)).values
es_mismo_cliente_2 = (df['numero_de_cliente'] == df['numero_de_cliente'].shift(2)).values

for col in top_vars:
    if col in df.columns:
        val_actual = df[col].values
        val_lag1 = df[col].shift(1).values
        val_lag2 = df[col].shift(2).values

        df[f'{col}_lag1'] = np.where(es_mismo_cliente_1, val_lag1, np.nan).astype(np.float32)
        df[f'{col}_lag2'] = np.where(es_mismo_cliente_2, val_lag2, np.nan).astype(np.float32)
        df[f'{col}_delta1'] = np.where(es_mismo_cliente_1, val_actual - val_lag1, np.nan).astype(np.float32)
        df[f'{col}_delta2'] = np.where(es_mismo_cliente_2, val_actual - val_lag2, np.nan).astype(np.float32)

df_train = df[df['foto_mes'].isin([202103, 202104, 202105, 202106])].copy()
del df; gc.collect()

mask_tr = df_train['foto_mes'] <= 202105
mask_val = df_train['foto_mes'] == 202106

y_val_baja2 = (df_train.loc[mask_val, 'clase_ternaria'] == 'BAJA+2').astype(np.int8).values
y_train = df_train.pop('clase_peligro').fillna(0).astype(np.int8)
df_train.drop(columns=cols_basura, inplace=True, errors='ignore')

X_tr_opt, y_tr_opt = df_train[mask_tr], y_train[mask_tr]
X_val_opt = df_train[mask_val]

def objective_low(trial):
    params = {
        'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt', 'is_unbalance': True,
        'random_state': 17 + trial.number, 'n_jobs': 2, 'verbose': -1, 'max_bin': 31,
        'min_data_in_leaf': 0, 'learning_rate': 0.005, 'device': 'gpu',
        'n_estimators': trial.suggest_int('n_estimators', 1500, 3000),
        'num_leaves': trial.suggest_int('num_leaves', 12, 35),
        'min_sum_hessian_in_leaf': trial.suggest_float('min_sum_hessian_in_leaf', 0.01, 5.0, log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.3, 0.6),
        'feature_fraction_bynode': trial.suggest_float('feature_fraction_bynode', 0.15, 0.4),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.7, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 5)
    }
    try:
        model = LGBMClassifier(**params)
        model.fit(X_tr_opt, y_tr_opt)
    except Exception:
        params.pop('device', None)
        model = LGBMClassifier(**params)
        model.fit(X_tr_opt, y_tr_opt)

    prob_val = model.predict_proba(X_val_opt)[:, 1]
    orden = np.argsort(prob_val)[::-1]
    y_val_ordenado = y_val_baja2[orden]
    return np.sum(np.where(y_val_ordenado[:11500] == 1, 1072500, -27500))

db_path_low = os.path.join(run_base_path, 'optuna_low.db')
study_low = optuna.create_study(direction='maximize', study_name='Optuna_Low', storage=f'sqlite:///{db_path_low}', load_if_exists=True)
study_low.optimize(objective_low, timeout=1200, n_jobs=4)

top_low = study_low.trials_dataframe()
top_low = top_low[top_low['state'] == 'COMPLETE'].sort_values(by='value', ascending=False)
print("\n=== PARÁMETROS DEL LOW TOP 2 SELECCIONADO ===")
print(top_low.iloc[1])