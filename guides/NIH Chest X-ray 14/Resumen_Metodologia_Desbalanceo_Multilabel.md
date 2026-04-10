# 📋 Resumen: Estrategias contra el Desbalanceo en Clasificación Multilabel de Rayos X
> **Fuente:** *"Deep Learning for Understanding Multilabel Imbalanced Chest X-Ray Datasets"*  
> Liz et al., 2022 — arXiv:2207.14408  
> **Dataset:** PadChest — 160,868 imágenes, 174 etiquetas, 67,625 pacientes

---

## 1. 🔥 El Problema: ¿Qué tan grave era el desbalanceo?

El dataset PadChest tiene un desbalanceo **extremo y doble**:

| Característica | Detalle |
|---|---|
| Clase **dominante** (`Normal`) | 47.4% del dataset (>35,000 muestras) |
| Clase **minoritaria** (`Supra aortic elongation`) | 0.28% (200 muestras) |
| **Ratio de imbalance** | 1:172 entre clase más frecuente y menos frecuente |
| Característica especial | Es **multilabel**: una imagen puede tener 12 enfermedades a la vez |
| El 6 top de clases ocupa | 82.7% del dataset completo |

> [!CAUTION]
> El problema no es solo que una clase tenga pocos ejemplos, sino que al ser **multilabel**, una imagen con patologías raras también tiene patologías comunes, complicando enormemente cualquier estrategia de balanceo.

---

## 2. 🛠️ Estrategias Aplicadas para Tratar el Desbalanceo

### 2.1 — Selección de Etiquetas (Umbral mínimo)

**El truco más simple y fundamental:** eliminar las clases con muy pocos ejemplos antes de entrenar.

- **Umbral fijado:** mínimo **200 muestras** por clase (0.22% del total)
- Si una imagen **solo** tenía etiquetas eliminadas, la imagen también se eliminaba
- **Resultado:** de 174 clases → 35 clases específicas o 54 clases generales (agrupando)

> [!NOTE]
> Esto no mejora el desbalanceo directamente, pero evita que el modelo intente aprender de clases con tan pocos ejemplos que cualquier aprendizaje sería ruido puro.

### 2.2 — Agrupación Jerárquica de Etiquetas

Los autores diseñaron **dos experimentos** para lidiar con el extremo número de clases:

| Experimento | Clases | Muestras | Precisión |
|---|---|---|---|
| **Clases Específicas** | 35 | 85,367 | Alta (señales radiológicas exactas) |
| **Clases Generales** | 54 | 90,687 | Menor (agrupa señales similares del árbol de términos) |

Al agrupar clases similares (e.g., todos los tipos de fractura → "Fractura") se aumenta el número de muestras por clase, aliviando el desbalanceo indirectamente.

### 2.3 — Preprocesamiento: Cropping por Segmentación Pulmonar (U-Net) ⭐ Clave

Este fue el **paso más crítico**. Sin él, 3 de 5 modelos no aprendían nada (AUC = 0.5).

El pipeline tiene 3 pasos:

```
Imagen original (RAW)
        ↓
[1] Generar máscara pulmonar con U-Net
        ↓
[2] Post-procesar máscara (flood fill, eliminar máscaras pequeñas, separar pulmones pegados)
        ↓
[3] Recortar imagen usando coordenadas de máscara + borde inferior
        ↓
Imagen final: Solo área de interés clínico (pulmones + área sub-pulmonar)
```

**¿Por qué ayuda con el desbalanceo?** Al eliminar zonas irrelevantes (brazos, cuello, fondo), el modelo puede aprender más rápido las señales sutiles de enfermedades raras con pocos ejemplos.

> [!IMPORTANT]
> Los resultados sin segmentación vs con segmentación muestran una diferencia brutal:
> - **Sin segmentación:** AUC global = 0.558 (solo DenseNet y EfficientNet aprendían algo)
> - **Con segmentación:** AUC global = 0.831 (ensemble CTP)

### 2.4 — Data Augmentation

Se aplicó un conjunto de transformaciones para generar variedad artificial, especialmente útil para clases minoritarias:

| Augmentación | Valor |
|---|---|
| Shear range | 0.1 |
| Zoom range | 0.1 |
| Rotation range | ±45° |
| Width/Height shift | 0.1 |
| Horizontal flip | ✅ |
| Brightness range | 0.7 – 1.1 |
| Channel shift | 0.05 |
| Fill mode | `nearest` |

**Impacto medido:**
- Sin data augmentation + con segmentación: AUC = 0.831
- **Con data augmentation + segmentación: AUC = 0.840**

### 2.5 — Función de Pérdida con Pesos de Clase ⭐ Clave

La función de pérdida utilizada fue **Weighted Cross-Entropy with Logits**:

$$\mathcal{L} = -\sum_{c=1}^{C} w_c \left[ y_c \log(\hat{y}_c) + (1 - y_c) \log(1 - \hat{y}_c) \right]$$

Donde $w_c$ es inversamente proporcional a la frecuencia de cada clase. Esto obliga a la red a **prestar más atención a los errores en clases raras**.

> [!TIP]
> A diferencia del enfoque del notebook ISIC2019 (que evitó el "double weighting"), este paper combina la pérdida ponderada directamente en la función de loss, sin un sampler adicional. En problemas multilabel esto es más robusto porque el sampler clásico es difícil de aplicar cuando hay múltiples etiquetas por muestra.

---

## 3. 🏗️ Arquitecturas Entrenadas

Se entrenaron 5 arquitecturas con **Transfer Learning desde ImageNet**:

| Modelo | Fortaleza | Estrategia de Fine-Tuning |
|---|---|---|
| **EfficientNet-B0** | Escalado eficiente de width/depth/resolution | Congelar primero 10% de capas conv. |
| **DenseNet-201** | Conexiones densas → reutilización de features, no hay vanishing gradient | Ídem |
| **InceptionV3** | Factorización de convoluciones + regularización por clasificador auxiliar | Ídem |
| **InceptionResNet-V2** | Combinación de Inception + ResNet (skip connections) | Ídem |
| **Xception** | Convoluciones depth-wise separables + skip connections | Ídem |

**Configuración común de entrenamiento:**

| Parámetro | Valor |
|---|---|
| Optimizer | Adam |
| Learning rate | 1e-4 |
| Batch size | 32 |
| Image size | 224×224 |
| Epochs (máx.) | 350 |
| Early stopping patience | 25 épocas |
| Early stopping threshold | 0.001 |
| Capas congeladas | Primeras 10% (detectan bordes/texturas básicas) |
| Cabeza clasificadora | 2 Dense layers (512 neuronas, ReLU, Dropout 0.2) |

---

## 4. 🧩 Estrategia Ensemble: El Diferenciador Principal

Se probaron tres métodos de combinación de los 5 modelos:

### CTP — Combine Then Predict ✅ **Ganador claro**
1. Cada modelo produce **probabilidades** (no predicciones binarias) para cada clase
2. Se calcula el **promedio de probabilidades** entre modelos
3. Se aplica un umbral para la decisión final

### PTC-lw — Predict Then Combine (Label-wise)
- Cada modelo hace su predicción binaria
- Se toma la **mayoría de votos por etiqueta** (independientemente de otras etiquetas)

### PTC-mode — Predict Then Combine (Mode)
- Cada modelo predice el **conjunto completo de etiquetas**
- Se selecciona el conjunto más frecuente entre los 5 modelos

> [!IMPORTANT]
> **CTP siempre supera a PTC**. La razón es que CTP preserva la información de incertidumbre (probabilidades) antes de combinar, mientras que PTC descarta esa información al binarizar primero.

---

## 5. 📊 Resultados: ¿Qué modelo fue el mejor?

### Clases Específicas (35 clases)

| Modelo | AUC Global | F1-score |
|---|---|---|
| **CTP Ensemble** | **0.840** | **0.647** |
| DenseNet-201 | 0.818 | 0.642 |
| EfficientNet-B0 | 0.804 | 0.629 |
| InceptionV3 | 0.797 | 0.625 |
| Xception | 0.782 | 0.625 |
| InceptionResNet-V2 | 0.780 | 0.621 |
| PTC-lw | 0.701 | 0.647 |
| PTC-mode | 0.677 | 0.637 |

### Clases Generales (54 clases)

| Modelo | AUC Global | F1-score |
|---|---|---|
| **CTP Ensemble** | **0.819** | **0.602** |
| EfficientNet-B0 | 0.767 | 0.600 |
| DenseNet-201 | 0.761 | 0.589 |
| InceptionResNet-V2 | 0.739 | 0.572 |
| Xception | 0.739 | 0.589 |
| InceptionV3 | 0.732 | 0.566 |

### Observación clave sobre clases minoritarias

> [!NOTE]
> **La metodología NO penaliza a las clases minoritarias.** Clases con <300 muestras (Hemidiaphragm elevation, Hiatal hernia, Sternotomy) lograron AUC > 0.93, igual o mejor que clases mayoritarias. Esto confirma que el conjunto de técnicas (segmentación + augmentation + weighted loss + ensemble) funcionó.

---

## 6. 🔑 ¿Qué Diferenció los Mejores Resultados? Jerarquía de Impacto

```
Impacto en AUC global (aprox.)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Segmentación pulmonar U-Net          +27.3 puntos AUC
   (sin seg: 0.558 → con seg: 0.831)
   ████████████████████████████████

2. Ensemble CTP vs mejor modelo individual  +2.2 puntos AUC
   (DenseNet: 0.818 → CTP: 0.840)
   ████

3. Data Augmentation                    +0.9 puntos AUC
   (sin aug+con seg: 0.831 → con aug: 0.840)
   ██

4. Weighted Loss (vs no balanced)      Esencial para no colapsar
   ▓ (sin esto, la mayoría de modelos no aprende)
```

### Resumen ejecutivo de qué hizo la diferencia:

| Técnica | ¿Necesaria? | Impacto |
|---|---|---|
| **Segmentación U-Net + Cropping** | ✅ Crítica | Sin ella, 3/5 modelos tienen AUC=0.5 (azar) |
| **Weighted Cross-Entropy with Logits** | ✅ Crítica | Permite aprender clases raras |
| **Ensemble CTP** | ✅ Muy recomendada | Siempre supera al mejor modelo individual |
| **Data Augmentation** | ✅ Importante | Mejora generalización, especialmente en clases raras |
| **Filtro mínimo de muestras (≥200)** | ✅ Necesaria | Elimina ruido de clases con muy poca data |
| **Agrupación jerárquica de clases** | ⭕ Opcional | Permite clasificar más clases a costa de precisión |
| **Transfer Learning ImageNet** | ✅ Estándar | Base de todos los modelos preentrenados |

---

## 7. 💡 Lecciones Aprendidas y Aplicabilidad al Proyecto

### Lo que este paper aporta a nuestro contexto (NIH ChestX-ray 14):

1. **La segmentación pulmonar es más importante que el modelo en sí.** Nuestro notebook NIH ya implementó `LungUNet` para esto — es la decisión correcta.

2. **Para multilabel, el Weighted Cross-Entropy es preferible a Focal Loss.** El paper confirma que `BCEWithLogitsLoss` con pesos de clase es el estándar en multilabel.

3. **El ensemble CTP siempre gana.** Si en producción se entrenan varios modelos (MaxViT, ConvNeXt, etc.), combinar sus probabilidades promedio dará el mejor resultado.

4. **El "double weighting" (sampler + weighted loss) puede ser problemático,** como vimos en el notebook ISIC. En multilabel, es mejor usar solo la pérdida ponderada.

5. **Las clases raras NO necesariamente obtienen peor AUC** si el pipeline está bien diseñado. El modelo puede aprender bien incluso con 200-300 muestras si el preprocesamiento es correcto.

### Comparación con nuestro notebook NIH:

| Aspecto | Paper (PadChest) | Nuestro Notebook (NIH) |
|---|---|---|
| Desbalanceo abordado con | Weighted Cross-Entropy | BCEWithLogitsLoss + Geometric Mean Sampler |
| Segmentación | U-Net (custom, post-procesada) | LungUNet (MobileNetV2 encoder) |
| Número de clases | 35–54 | 14 |
| Ensemble | 5 modelos (CTP) | 1 modelo (MaxViT-Tiny) |
| Data Augmentation | Extensa (rotación, zoom, etc.) | Mínima (solo flip horizontal) |
| Split | Estratificado por clase y paciente | Por PatientID (sin fuga) |

---

*Documento generado para el Proyecto 2 — Expert NIH Chest X-ray | Abril 2026*
