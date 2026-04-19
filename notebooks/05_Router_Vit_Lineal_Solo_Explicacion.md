# Explicacion tecnica detallada del `05_Router_Vit_Lineal_Solo.ipynb`

Este documento describe con mayor detalle la arquitectura y la logica operacional del notebook `05_Router_Vit_Lineal_Solo.ipynb`, incluyendo:

- preprocesamiento multimodal (2D y 3D),
- construccion de embeddings CLS,
- enrutamiento y balanceo de carga,
- definicion exacta de losses y su efecto practico,
- mecanismos ante asignaciones "incorrectas" de modalidad,
- estrategia de entrenamiento, checkpoints y gestion de VRAM.

---

## 1) Flujo general del sistema (vision de punta a punta)

El pipeline combina un **router ViT** con **5 expertos** (2D y 3D) especializados por dominio:

1. Se construye un lote mixto de muestras de distintos datasets.
2. El `AdaptivePreprocessor` convierte cada archivo a tensor valido para 2D o 3D.
3. `SwitchablePatchEmbed` convierte cada muestra en tokens y crea una secuencia unificada con CLS.
4. ViT-Tiny procesa la secuencia y produce un embedding CLS de 192 dimensiones.
5. La cabeza lineal del router produce logits sobre 5 expertos.
6. Durante entrenamiento profesor, el top-1 del router decide que experto ejecutar para computar `L_task`.
7. La optimizacion usa `L_routing`, `L_task` (si aplica) y `L_aux`.

Objetivo practico: que el router no solo acierte el dominio de origen, sino que aprenda decisiones que mejoran el rendimiento diagnostico y mantengan uso saludable de expertos.

---

## 2) Preprocesamiento de imagenes 2D y volumenes 3D

### 2.1 `AdaptivePreprocessor`: normalizacion multimodal

El notebook define `AdaptivePreprocessor` para unificar entradas heterogeneas:

- detecta tipo de archivo/modalidad,
- aplica rutina 2D o 3D correspondiente,
- entrega tensores compatibles con router y expertos.

### 2.2 Ruta 2D

En la ruta 2D (radiografia, ISIC, etc.):

- se carga la imagen,
- se ajusta tamaño/escala a la entrada esperada,
- se normalizan intensidades,
- se adapta canal cuando hace falta (por ejemplo, expandir 1 canal a 3 en ciertos modelos).

Esto permite que muestras de datasets distintos entren por una interfaz comun.

### 2.3 Ruta 3D

En la ruta 3D (LUNA, Pancreas):

- se carga volumen medico,
- se reescala/resamplea a forma objetivo,
- se normaliza intensidad volumetrica,
- se conserva estructura espacial para que el embedding 3D no destruya contexto anatómico.

### 2.4 Filtros de calidad previos al loader

Antes del `DataLoader`, hay filtrado por dataset:

- NIH: filtrado por subconjunto de clases compatibles con experto 1.
- NIH/Osteo: filtrado por entropia (`ENTROPY_MIN`, `ENTROPY_MAX`).

Impacto: cambia la distribucion real de muestras elegibles. Esto afecta train, cache CLS y metricas de validacion.

---

## 3) Construccion de embeddings y token CLS

### 3.1 `SwitchablePatchEmbed`: doble embedding en un solo espacio

`SwitchablePatchEmbed` define dos patch embedders:

- `patch_embed_2d` para tensores 2D,
- `patch_embed_3d` para tensores 3D.

Ambos proyectan al mismo `embed_dim=192`, requisito clave para que una sola cabeza de router compare modalidades distintas.

### 3.2 Conversion a secuencia y padding

Cada muestra produce una secuencia de longitud variable.

- `_patch_tokens_to_sequence(...)` convierte salidas de MONAI/timm a formato `[N_tokens, D]`.
- Se usa `pad_sequence(batch_first=True)` para construir batch rectangular.
- Se crea mascara booleana de tokens validos (vs padding).

### 3.3 Insercion de CLS y posicion

El router define:

- `cls_token` entrenable `[1,1,D]`,
- `pos_embed` entrenable `[1,513,D]`.

Para cada batch:

1. expande `cls_token` a tamaño batch,
2. concatena CLS + tokens de patch,
3. concatena mascara del CLS,
4. suma embedding posicional truncado a longitud actual.

### 3.4 ViT y salida de router

`VisionRouter` usa ViT-Tiny de `timm` con `patch_embed` reemplazado por `Identity`, porque los tokens ya vienen construidos.

Se pasan bloques transformer, luego:

- `x[:, 0]` = embedding CLS final (192),
- `router_head(cls)` = logits de enrutamiento sobre 5 expertos.

La salida del forward es `(logits_router, cls_token)`.

---

## 4) Loader, muestreo y balance de datos

### 4.1 Dataset mixto

`MixedMedicalDataset` devuelve:

- modo simple: `(tensor, dataset_id)`,
- modo profesor: `(tensor, dataset_id, task_label, path)`.

Esto habilita que la misma infraestructura sirva para routing puro o routing + task supervision.

### 4.2 `WeightedRandomSampler`

`build_router_dataloader_weighted` calcula pesos inversos a frecuencia por dominio.

Efecto:

- evita que un dataset masivo monopolice batches,
- aumenta probabilidad de ver dominios minoritarios.

Este balance es **de entrada** (data-level), distinto al balance de salida del gating.

---

## 5) Balanceo de carga del router (gating-level)

Se monitorean metricas por epoca:

- `route_pct`: porcentaje asignado a cada experto,
- `ratio = max(frac)/min(frac)`,
- `entropy`: entropia media de las probabilidades del router.

Interpretacion:

- **ratio alto** -> colapso (muy concentrado),
- **entropia muy baja** -> gating demasiado determinista,
- **entropia demasiado alta** -> router indeciso.

El mejor checkpoint usa un score que combina accuracy y penalizacion de desbalance:

`score = val_acc - 0.02 * max(0, val_ratio - 1.30)`.

---

## 6) Losses del sistema: definicion y rol practico

En `train_router_one_epoch`, la loss total es:

`L_total = L_routing + (L_task si existe) + alpha_aux * L_aux`

### 6.1 `L_routing` (CrossEntropy de enrutamiento)

- Entrada: logits del router.
- Target: `expert_ids` (dataset de origen).
- Funcion: aprendizaje base de asignacion dominio->experto.

Es la columna vertebral de estabilidad del router.

### 6.2 `L_task` (flujo profesor)

Se activa cuando hay:

- `experts` disponibles,
- `task_labels` validos en batch.

Procedimiento:

1. `pred_expert_ids = argmax(probs_router)`.
2. Agrupa muestras por experto elegido.
3. Ejecuta forward del experto correspondiente.
4. Pondera logits del experto por `gating_score` de esa muestra.
5. Calcula CE con etiqueta de tarea.
6. Promedia sobre expertos activos del batch.

Esto empuja al router hacia decisiones con utilidad diagnostica, no solo coincidencia de dominio.

### 6.3 `L_aux` (Switch-style)

`_switch_aux_loss(router_probs, num_experts)` implementa:

- fraccion hard de asignaciones por experto (`f_i`),
- probabilidad media por experto (`P_i`),
- combinacion tipo `N * sum_i f_i * P_i`.

Promueve reparto mas uniforme del trafico entre expertos.

### 6.4 Fix importante del notebook

El codigo fija:

`loss = L_routing + (L_task if L_task is not None else 0.0) + alpha_aux * aux`

Es decir, `L_routing` siempre presente. Esto evita que una señal de `L_task` dominante lleve a colapso de expertos.

---

## 7) Que pasa si se enruta "mal" una modalidad (2D->experto 3D o viceversa)

No existe una penalizacion hard-coded tipo "if modalidad incorrecta entonces +K". El castigo emerge por gradiente:

- sube `L_task` si el experto elegido no resuelve bien la tarea,
- sube `L_routing` si se aleja del experto esperado por dataset,
- sube `L_aux` si esa decision produce concentracion.

### Salvaguardas especificas en codigo

- NIH (multietiqueta) no participa en `L_task` bajo CE single-label (`_SINGLE_LABEL_DATASETS`).
- Conversion especial LUNA:
  - si experto LUNA requiere 3 canales y tensor llega 1 canal volumetrico, se usa `luna_1ch_to_kinetics_3ch`.
- Validaciones de etiquetas:
  - se descartan etiquetas no escalares o inconsistentes.

Resultado: no se "rompe" el batch por mismatch; el sistema intenta seguir entrenando con subconjuntos validos.

---

## 8) Entrenamiento por fases en el notebook 05

## 8.1 Fase 1 (base)

Configuracion destacada:

- `mode="head_only"`: entrena principalmente cabeza del router.
- `experts=experts_mgr.experts`.
- `warmup_epochs=2`.
- `expert_trainable=False`.

Dinamica:

- Epocas 1-2: solo `L_routing + L_aux` (sin feedback experto).
- Desde epoca 3: activa `L_task` (feedback profesor) con expertos congelados.

## 8.2 Fase 2 (finetune)

Configuracion:

- `mode="full"`,
- `expert_trainable=True`,
- `extra_param_groups` para cabezas de expertos.

Objetivo: ajustar conjuntamente router/backbone y partes de expertos para mejorar metrica final.

---

## 9) Uso de VRAM y eficiencia operacional

### 9.1 `ExpertsPinnedGPU`

Los expertos se cargan una vez y se "anclan" en GPU:

- reduce overhead de transferencias por batch,
- acelera entrenamiento profesor.

Costo: memoria base alta (todos los expertos residentes).

### 9.2 Congelado selectivo

Mecanismos para contener memoria/gradientes:

- `freeze_all_experts()`: evita gradiente en expertos cuando no se necesitan.
- `enable_head_finetune()`: habilita solo cabezas para ajuste ligero.
- modos de entrenamiento del router (`head_only`, `full`).

### 9.3 AMP y clipping

El loop usa:

- `torch.amp.autocast` en CUDA,
- `GradScaler`,
- `clip_grad_norm_`.

Beneficios:

- menor consumo efectivo de VRAM,
- estabilidad numerica ante gradientes grandes.

---

## 10) Checkpoints: que guardan y como se elige el mejor

`_save_router_ckpt(...)` guarda:

- `model_state_dict` completo,
- estado de `router_head`,
- mejor epoca y metadatos,
- historial (sin matrices de confusion completas para ahorrar peso).

Se guarda mejor checkpoint cada vez que mejora `score`.

Archivos principales del notebook:

- `router_professor_fase3_only_base.pth`,
- `router_professor_fase3_only_finetuned.pth`.

Esto habilita reanudacion y comparacion base vs finetune sin repetir todo el entrenamiento.

---

## 11) Por que puede haber gap train/val aunque no haya errores

En este notebook coexisten dos mundos:

- **train**: imagen/volumen -> preprocesador -> ViT vivo -> router,
- **val por epoca**: CLS cacheado en `.npz` -> `router_head`.

Si el cache CLS (`unified_cls_tokens.npz`) no corresponde al pipeline actual (cambios de filtros, timm, semilla, version, etc.), `val_acc` puede desviarse mucho de train.

Buenas practicas:

- regenerar cache CLS tras cambios de pipeline/config,
- mantener alineados filtros de dataset entre extraccion y entrenamiento,
- verificar distribucion por dominio (`route_pct`, `val_ratio`) junto con accuracy.

---

## 12) Resumen practico (reglas operativas)

1. Mantener preprocesamiento y filtros identicos entre extraccion CLS y train.
2. Revisar siempre `val_acc` junto a `val_ratio` y `val_entropy`.
3. Si hay colapso de experto, ajustar `alpha_aux`, warmup y/o balance de datos.
4. Usar `head_only` primero para estabilizar router antes de full finetune.
5. Regenerar `.npz` cuando cambien backbone, filtros o versiones de librerias.

Con esto, el notebook 05 funciona como una base robusta para los ablation notebooks (KAN, NeuralGMM, LogGaussianNB, SoftKNN) manteniendo la misma semantica de entrenamiento profesor.
