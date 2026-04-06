# Modelo de Propensión de Compra — Ferreycorp

## Descripción
Pipeline de Machine Learning para predecir si un cliente comprará o no un producto,
combinando segmentación K-Means, clasificadores supervisados con SMOTE, explicabilidad
SHAP y un agente conversacional basado en Claude para consultas en lenguaje natural.

---

## Estructura del proyecto

```
ml_project/
├── data/
│   ├── raw/              ← CSV original sin modificar
│   ├── processed/        ← Parquet limpio post-preprocesamiento
│   └── external/         ← Fuentes externas (APIs, scraping)
├── notebooks/            ← EDA y experimentos exploratorios
├── src/
│   ├── data/
│   │   ├── ingestion.py          ← Carga de datos
│   │   └── preprocessing.py      ← Outliers + limpieza
│   ├── features/
│   │   ├── feature_engineering.py ← Features de precio, promo, historial
│   │   └── feature_selection.py   ← Selección de columnas para modelo
│   ├── models/
│   │   ├── clustering.py          ← K-Means con K óptimo automático
│   │   ├── train.py               ← Entrenamiento + comparación de modelos
│   │   ├── predict.py             ← Generación del parquet de predicciones
│   │   └── tuning.py              ← RandomizedSearchCV
│   ├── evaluation/
│   │   ├── metrics.py             ← AUC, F1, Precision, Recall, CM
│   │   └── explainability.py      ← SHAP values + gráficos
│   ├── visualization/
│   │   ├── eda.py                 ← Distribuciones, correlaciones
│   │   └── model_plots.py         ← ROC, PR curve, Confusion Matrix
│   └── agent/
│       └── agent.py               ← Agente conversacional + feedback
├── models/
│   ├── trained/          ← model.pkl, clustering.pkl
│   └── experiments/      ← Runs de experimentos
├── outputs/
│   ├── predictions/      ← predicciones.parquet
│   ├── reports/images/   ← SHAP, ROC, CM, distribuciones
│   └── metrics/          ← metrics.json
├── config/
│   ├── config.yaml       ← Configuración general y rutas
│   └── model_params.yaml ← Hiperparámetros y modelo activo
├── tests/                ← Pruebas unitarias (pytest)
├── docs/                 ← Esta documentación
├── main.py               ← Pipeline completo de entrenamiento
├── requirements.txt
└── .env.example
```

---

## Inicio rápido

### 1. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 2. Configurar API key
```bash
cp .env.example .env
# Editar .env y agregar ANTHROPIC_API_KEY=sk-ant-...
```

### 3. Poner el CSV de datos
```
data/raw/compras_data.csv
```

### 4. Ejecutar el pipeline completo
```bash
python main.py
```

Con fine-tuning activado:
```bash
python main.py --tune
```

### 5. Lanzar el agente
```bash
python src/agent/agent.py
```

### 6. Correr tests
```bash
pytest tests/ -v
```

---

## Configuración

Todos los parámetros están en `config/`:

- **`config.yaml`**: rutas de archivos, columnas del dataset, configuración de outliers y clustering
- **`model_params.yaml`**: modelo activo (`active_model`), hiperparámetros por modelo, estrategia de threshold

Para cambiar de modelo, editar `config/model_params.yaml`:
```yaml
active_model: "gradient_boosting"   # random_forest | gradient_boosting | extra_trees | ...
```

---

## Agente conversacional

El agente responde preguntas en lenguaje natural sobre las predicciones:

```
Tú > ¿Cuántos clientes tienen prob > 70%?
Tú > Dame el top 10 del cluster 2 por propensión
Tú > Compara ingreso promedio entre quienes compran y no compran
Tú > ¿Qué variables influyen más en la decisión de compra?
```

Comandos especiales:
- `/limpiar` — reinicia la conversación
- `/feedback` — muestra estadísticas de feedback del equipo
- `/salir` — cierra el agente

### Sistema de feedback
Después de cada respuesta, el agente solicita un rating (1-5) y comentario opcional.
El feedback se guarda en `outputs/agent_feedback.jsonl` para análisis y mejora continua.

---

## Metodología

1. **Outliers**: IsolationForest (contamination=5%) + winsorizing P1-P99
2. **Clustering**: K-Means con K óptimo por Silhouette (rango 2-6)
3. **Features**: precio agregado (min, max, rango, std), actividad promocional, historial, interacciones cluster×precio
4. **Modelos**: Logistic Regression, Random Forest, Extra Trees, Gradient Boosting, SGD — todos con SMOTE para desbalance de clases
5. **Selección**: mejor modelo por Recall (minimiza falsos negativos = clientes con intención no captados)
6. **Threshold**: optimizado por Recall sobre validación
7. **Explicabilidad**: SHAP TreeExplainer para modelos de árbol, LinearExplainer para lineales

---

## Outputs generados

| Archivo | Descripción |
|---|---|
| `models/trained/model.pkl` | Pipeline serializado (scaler + SMOTE + clasificador) |
| `models/trained/clustering.pkl` | K-Means + scaler de segmentación |
| `outputs/predictions/predicciones.parquet` | 58k+ predicciones con prob, decil, cluster, features top |
| `outputs/metrics/metrics.json` | AUC, F1, Precision, Recall, Confusion Matrix |
| `outputs/reports/images/` | SHAP summary, ROC curve, PR curve, Confusion Matrix |
| `outputs/agent_feedback.jsonl` | Feedback del equipo sobre el agente |
