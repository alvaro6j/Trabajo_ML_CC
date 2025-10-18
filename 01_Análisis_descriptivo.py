# Importar las bibliotecas necesarias
import pandas as pd
import numpy as np

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

# Exploración inicial de los datos
print(desarrollo.info())
print("Primeras filas del conjunto de datos:")
print(desarrollo.head())

# Limpieza y preprocesamiento
# Verificar valores nulos
print("Valores nulos en cada columna:")
print(desarrollo.isnull().sum())

# Eliminar columna 'Id_Cliente':
desarrollo.drop('Id_Cliente', axis=1, inplace=True)

 # Conteo de registros duplicados:
print(f'Existen {desarrollo.duplicated().sum()} registros duplicados para tabla desarrollo.')

# Conteo de atributos duplicados:
duplicates = []
for col in range(desarrollo.shape[1]):
    contents = desarrollo.iloc[:, col]
    for comp in range(col + 1, desarrollo.shape[1]):
        if contents.equals(desarrollo.iloc[:, comp]):
            duplicates.append(comp)
duplicates = np.unique(duplicates).tolist()
print(f'Existen {len(duplicates)} atributos duplicados para tabla desarrollo.')
print(desarrollo.info())

# Eliminar registros duplicados:
desarrollo.drop_duplicates(inplace=True)
print(f'Existen {desarrollo.duplicated().sum()} registros duplicados para tabla desarrollo.')
desarrollo.describe(include='number').round(4)

# Dividir tablas de entrenamiento y validación:
data_train, data_test = train_test_split(desarrollo, test_size=0.3, random_state=0)
print(f'N° de registros entrenamiento: {data_train.shape[0]}')
print(f'N° de registros test: {data_test.shape[0]}')

# Descriptivos para todas las variables numéricas
print("\n📊 Estadísticas descriptivas (entrenamiento):")
print(data_train.describe(percentiles=[0.25, 0.5, 0.75]).round(4).T)

# Descriptivos para todas las variables numéricas donde 'Default' = 1
print("\n📊 Estadísticas descriptivas (entrenamiento):")
print(data_train[data_train['Default'] == 1].describe(percentiles=[0.25, 0.5, 0.75]).round(4).T)

# Descriptivos para todas las variables numéricas donde 'Default' = 0
print("\n📊 Estadísticas descriptivas (entrenamiento):")
print(data_train[data_train['Default'] == 0].describe(percentiles=[0.25, 0.5, 0.75]).round(4).T)

# Boxplot de variables independientes por categoría de 'Default' (0,1):
n_registros = data_train['Default'].value_counts()
print(n_registros)
print('-'*100)
variables = [col for col in data_train.select_dtypes(include='number').columns if col != "Default"]
print(f'Variables numéricas: {variables}')
print('-'*100)
n_cols = 3
n_rows = (len(variables) + n_cols - 1) // n_cols  # redondeo hacia arriba
fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
axes = axes.flatten() # Aplanar ejes para iterar fácilmente
categorias = data_train['Default'].unique() # Construir una paleta con tantos colores como categorías
palette = dict(zip(categorias, sns.color_palette("Set2", len(categorias))))
for i, col in enumerate(variables): # Dibujar los boxplots
    sns.boxplot(
        x="Default", y=col, data=data_train,
        ax=axes[i], hue="Default", legend=False, palette=palette
    )
    axes[i].set_title(f"Distribución de {col} según Default")
for j in range(len(variables), len(axes)): # Eliminar subplots vacíos si sobran
    fig.delaxes(axes[j])
plt.tight_layout()
plt.show()

# Pairplot
sample_data_train = data_train.sample(1000, random_state=22)
sns.pairplot(sample_data_train[['Edad', 'Años_Trabajando', 'Ingresos', 'Deuda_Comercial', 'Deuda_Credito', 'Otras_Deudas', 'Ratio_Ingresos_Deudas']])
plt.suptitle('Relaciones entre variables numéricas', y=1.02)
plt.show()

# Correlaciones
plt.figure(figsize=(8, 6))
sns.heatmap(sample_data_train.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm")
plt.title('Matriz de Correlación')
plt.show()

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

# # Imprimir resultados
print("=== Accuracy Promedio (CV=3) ===")
print(f"Logistic Regression:   {log_cv_acc:.4f}")
print(f"Random Forest:         {rf_cv_acc:.4f}")
print(f"HistGradientBoosting:  {hgb_cv_acc:.4f}")

# Seleccionar el mejor
candidates = [
    ("LogisticRegression", log_clf, log_cv_acc, None),
    ("RandomForest", rf_search.best_estimator_, rf_cv_acc, rf_search.best_params_),
    ("HistGradientBoosting", hgb_search.best_estimator_, hgb_cv_acc, hgb_search.best_params_)
]
best_name, best_model, best_cv_acc, best_params = sorted(candidates, key=lambda x: x[2], reverse=True)[0]
print("\n=== Mejor modelo seleccionado ===")
print(f"Modelo: {best_name}")
print(f"Accuracy CV: {best_cv_acc:.4f}")
if best_params:
    print("Mejores hiperparámetros:", best_params)

best_model.fit(X_train, y_train)
proba_test = best_model.predict_proba(X_test)[:,1]
thresholds = np.linspace(0,1,201)
accs = [accuracy_score(y_test, (proba_test>=t).astype(int)) for t in thresholds]
t_opt = thresholds[int(np.argmax(accs))]

y_pred_opt = (proba_test >= t_opt).astype(int)
print("\nUmbral óptimo:", round(t_opt,3))
print("Accuracy test:", accuracy_score(y_test, y_pred_opt))
print("AUC test:", roc_auc_score(y_test, proba_test))
print("\nMatriz de confusión:\n", confusion_matrix(y_test, y_pred_opt))
print("\nReporte:\n", classification_report(y_test, y_pred_opt))


# Gráficos para el mejor modelo

# Matriz de confusión
cm_hgb = confusion_matrix(y_test, y_pred_opt)
plt.figure(figsize=(6,4))
sns.heatmap(cm_hgb, annot=True, fmt='d', cmap='Greens')
plt.title('Matriz de Confusión - HistGradientBoosting')
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.show()

# Calcular y mostrar el ROC AUC Score
roc_auc_hgb = roc_auc_score(y_test, y_pred_opt)
print("ROC AUC Score para HistGradientBoosting:", roc_auc_hgb)

# Curva ROC
fpr_hgb, tpr_hgb, thresholds_hgb = roc_curve(y_test, y_pred_opt)
plt.figure(figsize=(6,4))
plt.plot(fpr_hgb, tpr_hgb, label='HistGradientBoosting (área = %0.3f)' % roc_auc_hgb)
plt.plot([0,1], [0,1], 'r--')
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva ROC - HistGradientBoosting')
plt.legend()
plt.show()

