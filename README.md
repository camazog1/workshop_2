# Workshop 2 – Machine Learning & Deep Learning Aplicado

**Universidad EAFIT – Introducción a la Inteligencia Artificial (2026-01)**

**Integrantes:** Kevin Quiroz & Carlos Mazo

---

## Descripción General

Este workshop integra dos problemas supervisados independientes desarrollados con el ciclo completo de un proyecto de Machine Learning y Deep Learning: análisis del problema, exploración de datos, preprocesamiento, entrenamiento, evaluación y análisis crítico de resultados.

---

## Estructura del Repositorio

```
workshop_2/
├── README.md
├── clasificacion/
│   └── clasificacion_emg.ipynb     ← Detección de fatiga muscular (EMG)
└── regresion/
    └── regresion.ipynb             ← Estimación de edad desde imágenes faciales
```

---

## Problema 1 – Clasificación: Detección de Fatiga Muscular en Ciclismo

### Dataset

| Campo | Detalle |
|---|---|
| **Nombre** | Muscle Fatigue Cycling |
| **Fuente** | HuggingFace – [YominE/Muscle_Fatigue_Cycling](https://huggingface.co/datasets/YominE/Muscle_Fatigue_Cycling) |
| **Señales** | EMG de 8 músculos de la pierna dominante durante sprints en bicicleta |
| **Frecuencia de muestreo** | 1000 Hz |
| **Target** | `0` = Condición normal · `1` = Desgaste muscular |

### Contenido del Notebook

El notebook `clasificacion_emg.ipynb` cubre las siguientes etapas:

1. **Análisis preliminar** – Recodificación del target a binario y clasificación de variables.
2. **Feature Engineering** – Extracción de 7 características por canal (4 en tiempo + 3 en frecuencia) sobre ventanas de 1 segundo (1000 muestras), generando un dataset de 56 features.
3. **EDA** – Distribuciones, correlaciones, boxplots por clase y análisis de balance.
4. **Preprocesamiento** – Pipeline de scikit-learn con estandarización y división 70/15/15.
5. **Entrenamiento y comparación** – kNN, Decision Tree, Random Forest, Gradient Boosting y DNN; ajuste de hiperparámetros con Random Search.
6. **Evaluación final** – Reentrenamiento del mejor modelo sobre `train + val`, análisis con matriz de confusión y métricas finales sobre `test`.
7. **Prueba con muestra artificial** – Generación de una muestra sintética e inferencia con el modelo seleccionado.

### Características Extraídas

| Dominio | Característica | Descripción |
|---|---|---|
| Tiempo | RMS | Energía promedio de la señal |
| Tiempo | Varianza | Dispersión de la amplitud |
| Tiempo | ZCR | Cruces por cero (proxy del contenido frecuencial) |
| Tiempo | MAV | Valor absoluto medio (nivel de activación) |
| Frecuencia | Frecuencia mediana | Indicador clásico de fatiga EMG |
| Frecuencia | Frecuencia media | Promedio ponderado del espectro |
| Frecuencia | Potencia espectral total | Energía total en el dominio frecuencial |

### Modelos Evaluados

- k-Nearest Neighbors (kNN)
- Decision Tree
- Random Forest
- Gradient Boosting
- Deep Neural Network (DNN) – mínimo 3 capas ocultas con Dropout y BatchNormalization

---

## Problema 2 – Regresión: Estimación de Edad a partir de Imágenes Faciales

### Dataset

| Campo | Detalle |
|---|---|
| **Nombre** | Faces: Age Detection from Images |
| **Fuente** | Kaggle – [arashnic/faces-age-detection-dataset](https://www.kaggle.com/datasets/arashnic/faces-age-detection-dataset) |
| **Entrada** | Imágenes faciales RGB |
| **Target** | Grupo de edad: `YOUNG` · `MIDDLE` · `OLD` |

### Contenido del Notebook

El notebook `regresion.ipynb` cubre las siguientes etapas:

1. **Análisis preliminar** – Justificación del enfoque, descripción de las imágenes y distribución de edades.
2. **EDA** – Histograma de clases, análisis de balance, visualización de muestras y calidad de imágenes.
3. **Preprocesamiento** – Redimensionamiento, normalización, data augmentation y división 70/15/15.
4. **Modelo CNN** – Arquitectura convolucional con regularización (Dropout, Batch Normalization); función de pérdida para regresión; métricas MAE, RMSE y R².
5. **Prueba con muestra artificial** – Inferencia sobre una imagen de prueba y análisis de sensibilidad.

---

## Requisitos

```bash
pip install datasets kagglehub scikit-learn tensorflow scipy matplotlib seaborn pandas numpy
```

> Los notebooks están preparados para ejecutarse en **Google Colab**. La primera celda de cada uno instala automáticamente las dependencias necesarias y descarga el dataset correspondiente.

---

## Métricas Reportadas

### Clasificación
| Métrica   | Train  | Val    | Test   |
|-----------|--------|--------|--------|
| Accuracy  | 1.0000 | 0.8844 | 0.8891 |
| Precision | 1.0000 | 0.8435 | 0.8522 |
| Recall    | 1.0000 | 0.7405 | 0.7481 |
| F1-Score  | 1.0000 | 0.7886 | 0.7967 |
### Regresión (CNN)

| Métrica | Train | Val | Test |
|---|---|---|---|
| MAE | 0.6546 | 0.2378 | 0.2281 |
| RMSE | 0.9036 | 0.5189 | 0.5134 |
| R² | -0.9898 | 0.3436 | 0.3582 |


---

## Referencias

- Dataset EMG: [YominE/Muscle_Fatigue_Cycling – HuggingFace](https://huggingface.co/datasets/YominE/Muscle_Fatigue_Cycling)
- Dataset imágenes: [arashnic/faces-age-detection-dataset – Kaggle](https://www.kaggle.com/datasets/arashnic/faces-age-detection-dataset)
- Phinyomark, A. et al. (2012). *Feature reduction and selection for EMG signal classification*. Expert Systems with Applications.
- Konrad, P. (2005). *The ABC of EMG*. Noraxon Inc.

> Esta documentación fue revisada y mejorada mediante IA.
