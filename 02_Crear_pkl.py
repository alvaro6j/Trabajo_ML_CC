# Importar las bibliotecas necesarias
import pandas as pd
import numpy as np
import pickle

# Para visualización de datos
import matplotlib.pyplot as plt
import seaborn as sns

# Para preprocesamiento y construcción del modelo
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, accuracy_score
from sklearn.utils.class_weight import compute_class_weight

# Cargar el conjunto de datos
desarrollo = pd.read_excel("C:/Users/alvaro.g/Documents/Cloud_Computing/Trabajo_grupal/Modelo_ML/Tabla_Trabajo_Grupal.xlsx")

# Eliminar columna 'Id_Cliente':
desarrollo.drop('Id_Cliente', axis=1, inplace=True)

# Eliminar registros duplicados:
desarrollo.drop_duplicates(inplace=True)

# Dividir tablas de entrenamiento y validación:
data_train, data_test = train_test_split(desarrollo, test_size=0.3, random_state=0)

# Separar variables de interés y variable objetivo:
X_train = data_train[['Edad', 'Años_Trabajando', 'Deuda_Comercial', 'Ratio_Ingresos_Deudas']]
y_train = data_train['Default']
X_test = data_test[['Edad', 'Años_Trabajando', 'Deuda_Comercial', 'Ratio_Ingresos_Deudas']]
y_test = data_test['Default']

# Escalamiento de características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# Entrenar un modelo de Regresión Logística con validación cruzada:
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
log_clf = LogisticRegression(max_iter=2000, class_weight='balanced', solver="liblinear")
log_clf.fit(X_train_scaled, y_train)
log_cv_acc = cross_val_score(log_clf, X_train_scaled, y_train, scoring="accuracy", cv=cv, n_jobs=-1).mean()

# Modelos RndomForest y HistGradientBoosting

SHEET_TRAIN = "Desarrollo"
TARGET_COL   = "Default"
ID_COLS      = ["Id_Cliente", "ID", "id", "id_cliente"]

df_train_full = pd.read_excel("C:/Users/alvaro.g/Documents/Cloud_Computing/Trabajo_grupal/Modelo_ML/Tabla_Trabajo_Grupal.xlsx")

assert TARGET_COL in df_train_full.columns, f"No se encontró la columna objetivo '{TARGET_COL}'."

y_full = df_train_full[TARGET_COL].astype(int)
X_full = df_train_full.drop(columns=[c for c in [TARGET_COL] + ID_COLS if c in df_train_full.columns])

# Detectar tipos
cat_cols_train = [c for c in X_full.columns if X_full[c].dtype == 'object' or str(X_full[c].dtype).startswith('category')]
num_cols_train = [c for c in X_full.columns if c not in cat_cols_train]

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_full, y_full, test_size=0.3, stratify=y_full, random_state=0
)

num_pre = Pipeline(steps=[
    ("imp", SimpleImputer(strategy="median")),
    ("scaler", RobustScaler())
])

cat_pre = Pipeline(steps=[
    ("imp", SimpleImputer(strategy="most_frequent")),
    ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

pre = ColumnTransformer(
    transformers=[
        ("num", num_pre, num_cols_train),
        ("cat", cat_pre, cat_cols_train)
    ],
    remainder="drop"
)


# Random Forest con búsqueda rápida:
rf_space = {                                # Espacios de búsqueda reducidos (para que sea más rápido)
    "clf__n_estimators": [150, 250, 350],
    "clf__max_depth": [None, 8, 12],
    "clf__min_samples_split": [2, 5, 10],
    "clf__min_samples_leaf": [1, 2, 4],
    "clf__max_features": ["sqrt", None]
}

rf_clf  = RandomForestClassifier(random_state=0, class_weight='balanced', n_estimators=400, n_jobs=-1)
pipe_rf  = Pipeline([("pre", pre), ("clf", rf_clf)])
rf_search = RandomizedSearchCV(
    estimator=pipe_rf,
    param_distributions=rf_space,
    n_iter=6,
    scoring="accuracy",
    cv=cv,
    random_state=0,
    n_jobs=-1,
    verbose=0
)
rf_search.fit(X_train, y_train)
rf_cv_acc = rf_search.best_score_

# HistGradientBoosting con búsqueda rápida:
hgb_space = {
    "clf__learning_rate": [0.05, 0.1],
    "clf__max_leaf_nodes": [31, 63],
    "clf__max_depth": [None, 5],
    "clf__l2_regularization": [0.0, 0.1]
}

hgb_clf = HistGradientBoostingClassifier(random_state=0)
pipe_hgb = Pipeline([("pre", pre), ("clf", hgb_clf)])
hgb_search = RandomizedSearchCV(
    estimator=pipe_hgb,
    param_distributions=hgb_space,
    n_iter=6,
    scoring="accuracy",
    cv=cv,
    random_state=0,
    n_jobs=-1,
    verbose=0
)
hgb_search.fit(X_train, y_train)
hgb_cv_acc = hgb_search.best_score_

# Seleccionar el mejor
candidates = [
    ("LogisticRegression", log_clf, log_cv_acc, None),
    ("RandomForest", rf_search.best_estimator_, rf_cv_acc, rf_search.best_params_),
    ("HistGradientBoosting", hgb_search.best_estimator_, hgb_cv_acc, hgb_search.best_params_)
]
best_name, best_model, best_cv_acc, best_params = sorted(candidates, key=lambda x: x[2], reverse=True)[0]
if best_params:
    print("Mejores hiperparámetros:", best_params)

best_model.fit(X_train, y_train)
proba_test = best_model.predict_proba(X_test)[:,1]
thresholds = np.linspace(0,1,201)
accs = [accuracy_score(y_test, (proba_test>=t).astype(int)) for t in thresholds]
t_opt = thresholds[int(np.argmax(accs))]

y_pred_opt = (proba_test >= t_opt).astype(int)

# Matriz de confusión
cm_hgb = confusion_matrix(y_test, y_pred_opt)

# Calcular y mostrar el ROC AUC Score
roc_auc_hgb = roc_auc_score(y_test, y_pred_opt)

# Curva ROC
fpr_hgb, tpr_hgb, thresholds_hgb = roc_curve(y_test, y_pred_opt)

# Guardar el modelo de regresión logística en un archivo .pkl
with open('modelo_LogisticRegression.pkl', 'wb') as archivo_salida:
    pickle.dump(log_clf, archivo_salida)

# Guardar el modelo RandomForest en un archivo .pkl
with open('modelo_RandomForest.pkl', 'wb') as archivo_salida:
    pickle.dump(rf_search.best_estimator_, archivo_salida)

# Guardar el modelo HistGradientBoosting en un archivo .pkl
with open('modelo_HistGradientBoosting.pkl', 'wb') as archivo_salida:
    pickle.dump(hgb_search.best_estimator_, archivo_salida)

# Si también deseas guardar el scaler
with open('scaler.pkl', 'wb') as archivo_salida:
    pickle.dump(scaler, archivo_salida)