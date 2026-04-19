# Preguntas clave de la consigna y preguntas probables (defensa + “cascarones”)

Fuente: [`consigna.md`](consigna.md), alineado al proyecto MoE médico (router ViT + ablation estadístico, 5 expertos, dashboard, reporte).

---

## 1. Pregunta científica explícita (núcleo del proyecto)

La consigna formula **una** pregunta central que el reporte y la demo deben responder con evidencia:

> **¿Justifica el Vision Transformer su costo computacional como router frente a métodos estadísticos clásicos operando sobre los mismos embeddings?**

- No asume que el ViT “debe ganar”: se valora el experimento, honestidad y discusión.
- Debe apoyarse en **mismos CLS / mismos Z** para los cuatro mecanismos (ablation justo).

---

## 2. Preguntas implícitas (objetivos de aprendizaje y secciones)

Equivalentes a “qué debe demostrar el equipo” según la consigna:

| Tema | Pregunta implícita |
|------|-------------------|
| MoE | ¿Cómo se evita que un solo experto reciba todo el tráfico (*expert collapse*) y cómo se mide? |
| Router | ¿Qué es *routing accuracy* y cómo se supervisa (etiqueta débil por dataset de origen)? |
| Ablation | ¿En qué difieren los cuatro mecanismos en tipo de parámetros, gradiente, VRAM y latencia? |
| Embeddings | ¿Por qué el backbone ViT está congelado para comparar cabezas sobre el mismo **z**? |
| GMM vs NB | ¿Cuándo `covariance_type='full'` es viable y cuándo conviene `diag` (y relación con NB)? |
| k-NN | ¿Por qué FAISS con IP tras L2-normalizar equivale a coseno? ¿Qué papel tiene la “maldición de la dimensionalidad” y PCA a 32? |
| Pérdida | ¿Qué papel cumple \(L_{\text{aux}}\) y cómo se elige \(\alpha\)? |
| Balance | ¿Qué significa el cociente \(\max(f_i)/\min(f_i)\) y el umbral **1.30**? |
| Hardware | ¿Por qué FP16, acumulación de gradientes y checkpointing en 3D? |
| Sin trampa | ¿Por qué **no** se puede pasar modalidad o metadatos como entrada (-20%)? |
| Datasets | ¿Cómo se unifica 2D vs 3D solo por **rank** del tensor (4 vs 5)? |
| Expertos | ¿Por qué **cinco** expertos heterogéneos y una arquitectura **distinta** por experto? |
| Expertos | ¿Qué pérdida y preprocesado usa cada dataset (NIH multilabel, ISIC CE, Osteo CE+CLAHE, 3D HU/Focal)? |
| Expertos | ¿Por qué gradient checkpointing es **obligatorio** en expertos 3D y qué implica en tiempo/VRAM? |
| Expertos | ¿Cómo entrenaron cada experto **antes** del MoE (Método A/B) y en qué fase se congelan respecto al router? |

---

## 3. Preguntas que el profesor puede hacer en defensa (por bloque)

### 3.1 Problema y motivación

- ¿Por qué un MoE con un solo tipo de imagen no basta para el enunciado?
- ¿Qué significa “enrutamiento sin conocer la modalidad de antemano” en la práctica de su implementación?

### 3.2 Preprocesado adaptativo

- ¿Cómo detectan 2D vs 3D sin metadatos? ¿Qué pasa si el tensor viene mal empaquetado?
- ¿Por qué NIH usa enfoque multietiqueta y LUNA binario a nivel scan según su pipeline?

### 3.3 Ablation study (Fases 0–2)

- ¿Por qué primero materializar/extraer **CLS** y luego entrenar/ajustar cabezas?
- ¿Linear+softmax vs GMM vs NB vs k-NN: en qué sentido el experimento es “justo”?
- Si k-NN empata o supera al ViT+Linear, ¿cómo lo interpretan? (la consigna lo anticipa.)

### 3.4 Expertos (datasets, arquitecturas y entrenamiento)

**Diseño y consigna**

- ¿Por qué la consigna exige **heterogeneidad** (arquitectura base distinta por experto) y no cinco copias del mismo modelo?
- ¿Qué relación hay entre “experto = especialista en un dominio/dataset” y la etiqueta débil de **routing** (origen del dato)?
- ¿Cómo mapean los **cinco expertos** a NIH, ISIC, Osteo, LUNA y Páncreas? ¿Hay mezcla de modalidades 2D/3D en la misma cola de entrenamiento del MoE?

**Arquitectura y justificación**

- Para NIH: ¿por qué ConvNeXt-Tiny (o la alternativa que usaron) frente a DenseNet-121 “histórico”? ¿Qué métrica miraron (F1 macro, por clase)?
- Para ISIC: ¿por qué EfficientNet-B3 o similar y qué rol tiene el **augmentation agresivo** y el **class weighting**?
- Para Osteoarthritis: ¿por qué VGG-16 BN / ResNet y el uso de **CLAHE** óseo?
- Para LUNA16: ¿por qué modelo 3D (R3D-18, ViViT, etc.) y cómo manejan **64³**, HU y checkpointing?
- Para Páncreas: ¿por qué FocalLoss y el problema de **imbalance severo**? ¿Swin3D vs R3D/MedicalNet?

**Entrenamiento y MoE**

- En el **Método B**: ¿en qué orden entrenaron expertos 2D, expertos 3D y router? ¿Cuántas épocas aproximadas por fase?
- ¿Los expertos están **congelados** cuando entrenan solo el router? ¿Cuándo los descongelan parcialmente (últimas capas) o hacen fine-tuning global?
- ¿Cómo construyen el **DataLoader mixto** “proporcional” para no sesgar el router hacia NIH (muchas imágenes) vs 3D (pocos volúmenes)?
- ¿Qué F1 macro reportan **por dataset** y cuál experto fue el más débil? ¿Cómo lo relacionan con el desbalance o el tamaño de dato?

**Inferencia**

- Tras el gating, ¿la salida final es solo la del experto activado o hay fusión/top-k? ¿Alineado con su implementación?
- ¿Cómo miden latencia por experto (3D vs 2D) en el dashboard o en experimentos?

### 3.5 Auxiliary loss y balance

- Escriban o expliquen \(L_{\text{aux}} = \alpha N \sum_i f_i P_i\): ¿qué parte es diferenciable y qué no?
- ¿Cómo calibraron \(\alpha\) y qué evolución vieron en \(f_i\)?

### 3.6 Métricas y umbrales

- F1 macro 2D vs 3D: ¿por qué umbrales distintos?
- Routing accuracy > 0.80: ¿en validación sobre qué split y con qué mezcla de modalidades?
- OOD: ¿cómo usan la entropía del gating y cómo fijaron el umbral?

### 3.7 Dashboard

- ¿Qué muestra el panel de ablation en vivo y cómo se actualiza?
- ¿La heatmap del router está alineada con la imagen que ve el usuario (misma prepro)?
- En el panel del **experto activado**: ¿muestran nombre, arquitectura y dataset de origen como pide la consigna?

### 3.8 Reporte (ABET / 7 páginas)

- ¿Cuál es la **contribución** del equipo (no repetir el enunciado)?
- ¿Qué limitación fue la más importante y cómo afectó los resultados?

### 3.9 Reproducibilidad

- ¿Qué seeds y `requirements` usan? ¿El clúster corre sin pasos manuales no documentados?

---

## 4. Preguntas “cascarón” (parecen simples; suelen pillar despreparación)

Estas son típicas de examen oral: respuesta corta pero exige definiciones precisas.

1. **“¿Su router es un clasificador de modalidad?”**  
   Trampa: la consigna prohíbe usar modalidad explícita; el router debe decidir solo con píxeles/embeddings. Respuesta correcta: aproxima qué experto (proxy de dominio), no una etiqueta de modalidad entregada por el usuario.

2. **“¿GMM con 5 componentes siempre es mejor que Naive Bayes?”**  
   Trampa: con pocos datos por clase, covarianza full puede singular; la propia consigna sugiere `diag` y relaciona con NB.

3. **“k-NN no tiene parámetros, entonces es más barato que GMM.”**  
   Trampa: el “modelo” es todo el train en memoria; coste de inferencia y RAM pueden ser mayores que un GMM ajustado.

4. **“Si el ViT+Linear tiene mejor routing accuracy, ¿por qué no usar solo eso y listo?”**  
   Trampa: la pregunta científica es **costo vs beneficio** (latencia, VRAM, interpretabilidad, estabilidad); el reporte debe discutir trade-offs, no solo el ganador.

5. **“¿La auxiliary loss entrena el balance?”**  
   Trampa: \(f_i\) es fracción dura (no diferenciable); la señal suave viene de \(P_i\). Conviene saber qué optimiza realmente el gradiente.

6. **“¿max/min < 1.30 aplica a todos los routers?”**  
   Trampa: la penalización **-40%** es solo para el router **ViT+Linear** del sistema final; los estadísticos no se penalizan igual (consigna explícita).

7. **“¿Por qué ImageNet norm en NIH y HU en LUNA?”**  
   Trampa: son tareas y dominios distintos; demuestra que entienden el prepro **por dataset**, no un único `transform` global ciego.

8. **“¿El ablation en sklearn usa GPU?”**  
   Trampa: la consigna dice que GMM, NB y k-NN van en **CPU**; GPU es sobre todo backbone + extracción de CLS.

### 4.1 Cascarones específicos de **expertos**

9. **“¿El mejor experto en F1 es el que más debe activar el router?”**  
   Trampa: el router optimiza **routing accuracy** (coincidir con el dataset de origen), no el F1 del experto; un experto muy fuerte en su tarea puede recibir poco tráfico si el sistema enruta mal.

10. **“¿Los cinco expertos compenetrados el mismo loss?”**  
   Trampa: NIH es **multilabel** (BCE), ISIC/Osteo **CE**, 3D con **Focal** en páncreas; mezclar sin ponderar mal puede sesgar el entrenamiento del MoE.

11. **“¿Por qué no usar un solo EfficientNet para los tres expertos 2D?”**  
   Trampa: la consigna exige **arquitectura base distinta** por experto (heterogeneidad); además la literatura por dataset difiere (tabla guía).

12. **“Expertos 3D sin checkpointing: ¿caben en 12 GB?”**  
   Trampa: la consigna marca **gradient checkpointing obligatorio** en 3D; sin eso es fácil superar VRAM o batch mínimo.

13. **“¿El experto LUNA predice nódulo o modalidad CT?”**  
   Trampa: en muchos pipelines del curso la tarea es **scan-level** u otra definición concreta; deben saber **qué etiqueta** entrenaron y cómo se relaciona con el “experto 4”.

---

## 5. Checklist rápido “¿estamos alineados con la consigna?”

- [ ] Pregunta científica respondida con **números** (tabla ablation + discusión).
- [ ] Cuatro mecanismos comparados sobre el **mismo Z** (mismo backbone congelado).
- [ ] Sin metadatos de modalidad en la ruta de inferencia del sistema final.
- [ ] Balance: cociente \(\max(f_i)/\min(f_i)\) reportado para el router ViT+Linear; estrategia si se acerca al límite.
- [ ] Dashboard: ablation + heatmap + OOD + load balance visibles.
- [ ] Reporte ≤ 7 páginas con figuras obligatorias y limitaciones honestas.
- [ ] **Expertos:** cinco modelos con **backbones distintos** (o justificación si hubo equivalente); checkpoints documentados; F1 por dataset; 3D con **checkpointing**; DataLoader mixto razonable para no sesgar el router.

---

*Documento auxiliar para preparación de defensa; no sustituye la lectura de [`consigna.md`](consigna.md).*
