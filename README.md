# Detección de Fraude Financiero con Machine Learning e IA Generativa

Sistema end-to-end de detección de fraude en transacciones con tarjeta de crédito, diseñado 
con foco en aplicabilidad real para equipos de riesgo bancario: no solo predice, sino que 
traduce el costo de negocio en decisiones de modelado, y explica cada alerta en lenguaje natural.

## Contexto de negocio

Los bancos enfrentan un trade-off constante entre dos tipos de error costosos:
- **Fraude no detectado**: pérdida del monto completo de la transacción.
- **Falsa alarma**: fricción con el cliente y costo de revisión manual.

Este proyecto no se limita a maximizar una métrica de machine learning — optimiza directamente 
el costo esperado en dólares para el negocio, y hace el resultado interpretable para analistas 
no técnicos mediante IA generativa.

## Dataset

[Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (Kaggle) 
— 284.807 transacciones de tarjetas de crédito europeas (septiembre 2013), con 492 casos de 
fraude (0.173%). Variables `V1`-`V28` anonimizadas vía PCA por confidencialidad.

**El dataset no se incluye en este repo** (150MB, excede buenas prácticas de Git). Para reproducir:
1. Descargalo desde el link de Kaggle.
2. Colocá `creditcard.csv` en la carpeta `data/`.

## Arquitectura del proyecto
Financiero/
├── data/                   # Dataset (no versionado)
├── notebooks/
│   └── 01_eda.ipynb        # EDA, modelado, SHAP, explicación con LLM
├── sql/                    # [COMPLETAR] Queries de análisis
├── powerbi/                # [COMPLETAR] Dashboard .pbix
├── src/
│   └── fraud_model.pkl     # Modelo entrenado serializado
└── README.md

## Metodología

### 1. Análisis exploratorio
Identificación de desbalanceo extremo de clases (0.173% fraude), distribución de montos por 
clase, y un patrón horario claro: el fraude se concentra en franjas de menor vigilancia 
(madrugada y pre-mediodía).

### 2. Split temporal (no aleatorio)
A diferencia del enfoque estándar con `train_test_split` aleatorio, se entrena con el 80% de 
transacciones más antiguas y se evalúa con el 20% más reciente — simulando el escenario real 
de producción, donde nunca se entrena con datos del futuro. Este enfoque reveló un *data drift*: 
la tasa de fraude cae de 0.183% (train) a 0.132% (test).

### 3. Modelado
Random Forest con `class_weight='balanced'` para manejar el desbalanceo sin generar datos 
sintéticos (se evitó SMOTE deliberadamente, ya que interpolar sobre variables ya proyectadas 
por PCA puede generar registros sin sentido físico real).

**Métricas:** PR-AUC = 0.81 (métrica robusta a desbalanceo, preferida sobre ROC-AUC y accuracy).

### 4. Optimización de threshold por costo de negocio
En vez de usar el umbral por defecto (0.5), se calculó el threshold que minimiza el costo total 
esperado, considerando el costo real de cada tipo de error (fraude no detectado vs. revisión 
manual).

**Corrección metodológica importante:** la primera iteración calculó el threshold óptimo sobre 
el mismo test set usado para evaluar — un data leakage sutil. Se corrigió separando un set de 
validation exclusivamente para la selección del threshold, evaluando una única vez sobre el 
test set intacto. El threshold cambió de 0.33 (con leakage) a 0.21 (correcto), confirmando que 
la primera estimación estaba parcialmente sobreajustada.

**Resultado final:** con threshold=0.21, el modelo reduce los fraudes no detectados de 19 a 12 
casos (sobre 75 fraudes reales en test), a costa de más falsas alarmas (de 8 a 77) — un 
trade-off justificado porque el costo de un fraude no detectado es ~20x mayor al costo de una 
revisión manual.

### 5. Explicabilidad con SHAP
Cada predicción se descompone en la contribución de cada variable, identificando V14, V10, V4, 
V12 y V17 como las de mayor impacto consistente. Esto no es solo un ejercicio técnico: en banca, 
la capacidad de explicar una decisión automatizada es un requisito de auditoría y compliance.

### 6. Explicación en lenguaje natural con IA generativa
Los valores SHAP de cada alerta se traducen a una explicación en español, sin jerga técnica, 
usando un LLM (Qwen2.5-7B vía Hugging Face Inference API). Esto cierra la brecha entre el 
modelo y el analista de riesgo que consume la alerta en su trabajo diario.

## Stack técnico

- **Python**: pandas, scikit-learn, SHAP, matplotlib/seaborn
- **IA Generativa**: Hugging Face Inference API (Qwen2.5-7B-Instruct)
- **SQL**: SQLite, window functions (RANK), CTEs, agregaciones — identificó una concentración 
anómala de falsos positivos en horario nocturno (21-22h, ~90% vs. ~45% en el resto del día)
- **Visualización**: Power BI

## Resultados clave

| Métrica | Valor |
|---|---|
| PR-AUC | 0.81 |
| Recall (fraude detectado) con threshold óptimo | 84% (63 de 75 casos) |
| Falsos negativos (fraude no detectado) | 12 de 75 (vs. 19 con threshold default) |
| Reducción de costo vs. threshold default | 18.8% (-$376,40 en 2 días de test) |

## Cómo reproducir

```bash
git clone [tu-repo-url]
cd Financiero
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# Descargar creditcard.csv de Kaggle y colocar en data/
jupyter notebook notebooks/01_eda.ipynb
```

## Autora

Daniela Andrea Triador — [LinkedIn](https://linkedin.com/in/dtriador) | [Portfolio](https://dtriador.github.io/DTriador)