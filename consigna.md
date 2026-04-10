

PROYECTO FINAL — BLOQUE VISIÓN
Clasificación Médica con MoE + Ablation Study del Router

Curso	Incorporar Elementos de IA — Unidad II (Bloque Visión)
Tema	Clasificación Médica Multimodal con Mixture of Experts
Duración	4 semanas (Semanas 9–12 del bloque)
Hardware disponible	Clúster universitario con 2 tarjetas gráficas de 12 GB de VRAM cada una
Dirigido a:	Pregrado Ingeniería de Datos / IA
Entregables	(1) Sistema MoE completo + Dashboard  (2) Reporte técnico 7 páginas máx. (ajustado a requisitos ABET)

 
1. Introducción y Relevancia del Proyecto

La inteligencia artificial aplicada a imágenes médicas representa uno de los campos de mayor impacto en la medicina moderna. El desafío real no es construir un modelo excelente para una sola tarea — es construir sistemas capaces de manejar múltiples tipos de imágenes médicas con una única infraestructura eficiente, sin conocer de antemano qué tipo de imagen recibirán.

Este proyecto construye un sistema de Mixture of Experts (MoE) donde el router — el componente que decide qué experto activa — es evaluado experimentalmente a través de cuatro mecanismos de diferente naturaleza matemática: uno de aprendizaje profundo (Vision Transformer) y tres de naturaleza estadística (GMM, Naive Bayes, k-NN). Esta comparación no es accesoria — es el núcleo científico del proyecto.

El reto central: el Router como objeto de estudio científico
El sistema recibe únicamente la imagen o volumen médico, sin ningún metadato ni etiqueta de modalidad.
El router debe aprender por sí solo — mirando solo los píxeles — a qué experto enviar cada caso.
La pregunta que el proyecto debe responder científicamente es:
  ¿Justifica el Vision Transformer su costo computacional como router frente a métodos estadísticos
   clásicos operando sobre los mismos embeddings?
La respuesta no es obvia y debe demostrarse con datos reales en el reporte técnico.

1.1 Objetivos de Aprendizaje
1.	Diseñar e implementar un Vision Transformer (ViT, CvT o Swin-Tiny) como router de enrutamiento automático.
2.	Implementar tres routers estadísticos (GMM, Naive Bayes, k-NN) como baselines comparativos sobre los mismos embeddings.
3.	Diseñar y ejecutar un ablation study formal comparando los cuatro mecanismos de routing.
4.	Implementar un pipeline de preprocesado adaptativo que redimensione imágenes 2D y volúmenes 3D automáticamente.
5.	Aplicar técnicas de manejo de VRAM: Gradient Accumulation, FP16, gradient checkpointing.
6.	Implementar la Auxiliary Loss de balanceo de carga (Switch Transformer) y calibrar α.
7.	Seleccionar y justificar las arquitecturas de expertos que den mejor desempeño.
8.	Redactar un reporte técnico de máximo 7 páginas con resultados, alcances y limitaciones.

 
2. Datasets y Preprocesado Adaptativo

El sistema recibe como única entrada la imagen o volumen médico. El pipeline de preprocesado detecta automáticamente si la entrada es 2D o 3D e interpola a las dimensiones requeridas sin que el modelo ni el usuario deban informar la modalidad.

Dataset	Modalidad	Clases	Volumen	Preprocesado requerido
NIH ChestX-ray14	Radiografía 2D	14 patologías	~112 K imgs	Resize 224×224. Normalización ImageNet. BCEWithLogitsLoss (multilabel).
ISIC 2019	Fotografía derm. 2D	9 clases	~25 K imgs	Resize 224×224. Augmentation agresivo. Class-weighted CrossEntropyLoss.
Osteoarthritis	Radiografía rodilla 2D	3 grados KL	~10 K imgs	Resize 224×224. CLAHE para contraste óseo. CrossEntropyLoss.
LUNA16 / LIDC-IDRI	CT pulmonar 3D	2 clases	~600 vol. (subset)	Resize 64×64×64. Normalización HU [-1000,400]. Gradient checkpointing obligatorio.
Pancreatic Cancer	CT abdominal 3D	2 clases	~281 vol.	Resize 64×64×64. FocalLoss (imbalance severo). Gradient checkpointing obligatorio.

⚠  Sin metadatos — única entrada: la imagen
El sistema NO recibe texto, etiquetas de modalidad, nombres de archivo ni ningún metadato como entrada.
La detección 2D/3D es automática por la forma del tensor (rank=4 para 2D, rank=5 para 3D).
Soluciones que pasen la modalidad como entrada adicional son penalizadas con -20% de la nota.

 
3. Arquitectura del Sistema MoE

┌──────────────────────────────────────────────────────────────────┐
│  Imagen o Volumen (PNG / JPEG / NIfTI)  —  sin metadatos        │
└───────────────────────────┬──────────────────────────────────────┘
                            │
         ┌──────────────────▼──────────────────────┐
         │     PREPROCESADOR ADAPTATIVO            │
         │  rank=4 → 2D resize 224×224             │
         │  rank=5 → 3D resize 64×64×64            │
         │  → tokens [B, N, d_model]               │
         └──────────────────┬──────────────────────┘
                            │
         ┌──────────────────▼──────────────────────┐
         │    BACKBONE COMPARTIDO (ViT congelado)  │
         │    Patch Embedding + Self-Attention      │
         │    → CLS token z ∈ R^d_model           │
         └──────────────────┬──────────────────────┘
                            │  mismo z para los 4 routers
    ┌───────────────────────┼──────────────────────────────┐
    ▼                       ▼             ▼                ▼
┌──────────┐      ┌──────────────┐  ┌──────────┐  ┌──────────┐
│ViT+Linear│      │  ViT + GMM   │  │ViT+NaiveB│  │ViT+ k-NN│
│(baseline │      │  (param.est.)│  │(param.est│  │(no-param)│
│   DL)    │      │              │  │          │  │  FAISS   │
└────┬─────┘      └──────┬───────┘  └────┬─────┘  └────┬────┘
     └──────────────────────────────────────────────────┘
                            │  ABLATION STUDY
                            │  → Routing Accuracy por método
                            │  → El mejor método continúa
                            ▼
         ┌──────────────────────────────────────────┐
         │   5 EXPERTOS HETEROGÉNEOS (a elección)  │
         │   Exp1(2D) Exp2(2D) Exp3(2D)            │
         │   Exp4(3D) Exp5(3D) — grad.checkpoint   │
         └──────────────────┬───────────────────────┘
                            │
         ┌──────────────────▼───────────────────────┐
         │  L = L_task + α·L_aux  (Switch Transf.)  │
         └──────────────────────────────────────────┘

 
4. Ablation Study del Router: Cuatro Mecanismos en Competencia

Esta sección define el experimento central del proyecto. Los cuatro mecanismos de routing compiten sobre exactamente el mismo backbone ViT congelado, usando el mismo CLS token como representación de entrada. El experimento es justo porque lo único que varía es la cabeza de decisión.

4.1 Diseño del Experimento
FASE 0: Extraer CLS tokens
  Entrenar backbone ViT (o cargar preentrenado timm)
  Pasar todo el dataset → guardar CLS tokens en disco
  Resultado: Z_train [N_train, d_model], Z_val [N_val, d_model]
  con etiquetas de experto correcto (supervisión débil por dataset de origen)

FASE 1: Comparar 4 cabezas de routing sobre Z_train / Z_val
  A) Linear + Softmax  → entrenamiento con CrossEntropy + gradiente
  B) GMM (5 comp.)     → ajuste con EM (sklearn GaussianMixture)
  C) Naive Bayes       → ajuste con MLE (sklearn GaussianNB)
  D) k-NN (k=5)        → indexación FAISS, distancia coseno

FASE 2: Seleccionar el router ganador (mayor Routing Accuracy en validación)
  → Integrar en el sistema MoE completo
  → Reportar en el reporte técnico con tabla comparativa

4.2 Los Cuatro Mecanismos de Routing

Mecanismo A — ViT + Linear + Softmax (Baseline de Deep Learning)
La cabeza de routing es una capa lineal entrenada end-to-end con el backbone:
g = softmax(W_gate · z + b)  ∈ R^5
donde z = CLS token ∈ R^d_model, W_gate ∈ R^(5 × d_model)
Entrenado con CrossEntropyLoss + Noisy Top-K Gating
Auxiliary Loss: L_aux = α · N · Σ f_i · P_i
# Implementación
class LinearGatingHead(nn.Module):
    def __init__(self, d_model, n_experts):
        super().__init__()
        self.gate = nn.Linear(d_model, n_experts)
    def forward(self, z):
        return F.softmax(self.gate(z), dim=-1)  # [B, n_experts]

Mecanismo B — ViT + GMM (Paramétrico Estadístico)
Un Gaussian Mixture Model con 5 componentes, uno por experto. Cada componente modela la distribución de los CLS tokens de su dominio médico. La asignación es por probabilidad posterior de Bayes:
P(experto_i | z) ∝ π_i · N(z ; μ_i, Σ_i)
donde π_i = peso del componente i
      μ_i = media de CLS tokens del experto i
      Σ_i = covarianza (full, tied, diag o spherical)
Entrenado con EM sobre Z_train — sin gradiente descendente
from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(
    n_components=5,
    covariance_type='full',   # probar también 'diag' si OOM
    max_iter=200,
    random_state=42
)
gmm.fit(Z_train)             # Z_train: [N, d_model] — numpy array
probs = gmm.predict_proba(Z_val)  # [N_val, 5]
routing_pred = probs.argmax(axis=1)

💡  Nota sobre d_model y covarianza full
Con d_model=192 (ViT-Tiny), la matriz de covarianza full tiene 192² = 36,864 parámetros por componente.
Con 5 componentes: ~184K parámetros. Cabe perfectamente en CPU.
Si el dataset de entrenamiento tiene menos de ~5000 muestras por experto, usar covariance_type='diag'
para evitar matrices singulares. En ese caso el GMM equivale matemáticamente a Naive Bayes.

Mecanismo C — ViT + Naive Bayes (Paramétrico Simple)
Gaussian Naive Bayes: asume independencia entre las d_model dimensiones del CLS token. La matriz de covarianza es diagonal — una varianza por dimensión. Mucho más rápido que GMM y sorprendentemente competitivo cuando el backbone ya organiza bien el espacio de embeddings:
P(experto_i | z) ∝ P(experto_i) · ∏_{j=1}^{d} P(z_j | experto_i)
donde P(z_j | experto_i) = N(z_j ; μ_{ij}, σ²_{ij})
Supuesto: dimensiones del embedding son independientes dado el experto
MLE: μ_{ij} = media de z_j en ejemplos del experto i
      σ²_{ij} = varianza de z_j en ejemplos del experto i
from sklearn.naive_bayes import GaussianNB

# y_train: etiqueta del experto correcto (0–4)
# derivada del dataset de origen de cada imagen
nb = GaussianNB()
nb.fit(Z_train, y_train_expert)
probs = nb.predict_proba(Z_val)   # [N_val, 5]
routing_pred = probs.argmax(axis=1)

Importante: Naive Bayes es paramétrico. Su clasificación 'no paramétrica' se refiere a que no requiere optimización iterativa (la solución es analítica en forma cerrada), no a que no tenga parámetros. Los parámetros son μ_{ij} y σ²_{ij} estimados directamente de los datos.

Mecanismo D — ViT + k-NN con FAISS (No Paramétrico)
k-NN es el único método verdaderamente no paramétrico del ablation: no asume ninguna forma distribucional y su 'memoria' son los datos de entrenamiento mismos. Cada imagen nueva se enruta al experto más votado por sus k vecinos más cercanos en el espacio de embeddings:
router(z) = argmax_i Σ_{j ∈ kNN(z)} 1[experto(j) = i]
Distancia: similitud coseno  sim(z, z_j) = (z · z_j) / (||z|| · ||z_j||)
No hay parámetros que aprender — el modelo son los N_train embeddings guardados
Complejidad de inferencia: O(N_train · d) sin indexación
                           O(log N_train) con índice FAISS IVF
import faiss, numpy as np

# Construcción del índice (una vez, en entrenamiento)
d = Z_train.shape[1]  # d_model, e.g. 192
# Normalizar para usar producto interno como coseno
faiss.normalize_L2(Z_train)
index = faiss.IndexFlatIP(d)  # Inner Product = coseno normalizado
index.add(Z_train)

# Inferencia
faiss.normalize_L2(Z_query)
D, I = index.search(Z_query, k=5)   # D: distancias, I: índices
# Voto por mayoría de los k vecinos
neighbors_labels = y_train_expert[I]  # [B, k]
routing_pred = np.apply_along_axis(
    lambda x: np.bincount(x, minlength=5).argmax(),
    axis=1, arr=neighbors_labels
)

⚠  Maldición de la dimensionalidad y k-NN
Con d_model=192, la búsqueda exhaustiva sobre 112K embeddings de ChestX-ray14 requiere ~85 MB en float32.
Esto es perfectamente manejable en CPU con FAISS IndexFlatIP.
Sin embargo, k-NN SIN reducción de dimensión puede sufrir la maldición de la dimensionalidad:
en espacios de alta dimensión, todas las distancias tienden a valores similares.
Solución recomendada si k-NN no converge: aplicar PCA a d=32 antes del índice FAISS.
Reportar en el reporte técnico si fue necesario y el impacto en Routing Accuracy.



4.3 Tabla Comparativa Esperada (Guía de Resultados)
Esta tabla muestra rangos esperados basados en la literatura. Sus resultados reales pueden diferir — lo importante es reportarlos honestamente y discutir por qué difieren:

Router	Tipo	Routing Acc.	Latencia	Parámetros	Gradiente	VRAM
ViT + Linear (baseline DL)	Paramétrico (gradiente)	~85–92%	~8 ms	~5.7 M	Sí	~1.5 GB
ViT + GMM	Paramétrico (EM)	~78–87%	~3 ms	5×(d+d²)	No	~50 MB
ViT + Naive Bayes	Paramétrico (MLE analít.)	~72–82%	~1 ms	5×2d	No	~5 MB
ViT + k-NN (FAISS coseno)	No paramétrico	~75–85%	~5 ms (con índice)	N×d (todos train)	No	~85 MB

💡  ¿Qué resultado es 'correcto'?
No hay un resultado correcto predefinido. La hipótesis natural es que ViT+Linear sea el mejor,
pero la literatura reciente muestra que métodos estadísticos sobre buenos embeddings pueden ser
competitivos o incluso superiores en Routing Accuracy cuando los datos son limitados.
Lo que se evalúa es la calidad del experimento, la honestidad del análisis y la profundidad
de la discusión — no que el ViT gane.

 
5. Expert Load Balancing

El 'Expert Collapse' — el router enviando todo al mismo experto — destruye la especialización del sistema MoE. La Auxiliary Loss del paper Switch Transformer (Fedus et al., 2021) lo previene:

L_aux = α · N · Σ(i=1 to N)  f_i · P_i
f_i = fracción de imágenes enrutadas al experto i (dura, no diferenciable)
P_i = probabilidad media asignada al experto i (suave, diferenciable)
N = 5 expertos,  α ∈ [0.01, 0.1]
L_total = L_task + α · L_aux

Corrección importante sobre la notación del balance: el umbral de penalización usa un cociente, no un porcentaje. Con 5 expertos perfectamente balanceados todos tienen f_i = 0.20. Si el más cargado tiene f = 0.26 y el menos f = 0.20, el cociente es 0.26/0.20 = 1.30 — justo en el límite. Un cociente de 2.0 significa que un experto recibe el doble que otro.

⚠  Penalización del 40% — cociente max(f_i)/min(f_i)
Si al momento de la evaluación final:   max(f_i) / min(f_i)  >  1.30
→ Penalización del 40% sobre la nota total obtenida.

Ejemplo numérico: si Exp1 recibe 35% de imágenes y Exp5 recibe 10%:
  0.35 / 0.10 = 3.50  >>  1.30  →  penalización activa.

Ejemplo en el límite: Exp1=26%, Exp5=20%: 0.26/0.20 = 1.30 → justo en el umbral.
Nota: esta penalización solo aplica al router ViT+Linear. Los routers estadísticos
no tienen L_aux — su balance se mide pero no se penaliza de la misma forma.

 
6. Arquitecturas de Expertos (Opcionales y Justificadas)

Las arquitecturas de expertos son sugerencias. Los estudiantes eligen y justifican la que da mejor desempeño en cada dataset. La restricción es heterogeneidad: cada experto debe usar una arquitectura base distinta.

Experto	Dataset	Arquitectura guía	Alternativas que pueden dar mejor F1
Exp. 1	NIH ChestX-ray14	ConvNeXt-Tiny	DenseNet-121 (referencia histórica del dataset), ResNet-50, EfficientNet-B2
Exp. 2	ISIC 2019	EfficientNet-B3	EfficientNetV2-S, ConvNeXt-Small. EfficientNet es SOTA clásico para este dataset.
Exp. 3	Osteoarthritis	VGG-16 BN	ResNet-34, Inception-V3. La tarea es relativamente simple — modelos livianos suelen funcionar bien.
Exp. 4	LUNA16 CT 3D	ViViT-Tiny	R3D-18 o MC3-18 (torchvision.models.video) — más fáciles de implementar que ViViT.
Exp. 5	Pancreatic Cancer 3D	Swin3D-Tiny	R3D-18, MedicalNet (pretrained on medical CT). MedicalNet puede dar ventaja significativa.

7. Métodos de Entrenamiento

Método A — End-to-End
9.	Épocas 1–10: Congelen todos los expertos. Solo router + cabezas de clasificación. LR=1e-3.
10.	Épocas 11–25: Descongelen últimas 2 capas de expertos 2D. LR=1e-4.
11.	Épocas 26–40: Fine-tuning global. LR=1e-5. Expertos 3D siempre con gradient checkpointing.

Método B — Por Partes
12.	FASE 1 (S9): Expertos individualmente en sus datasets. 15–30 épocas según dataset.
13.	FASE 2 (S10): Router aislado con expertos congelados. DataLoader mixto proporcional. 30 épocas.
14.	FASE 3 (S11): Fine-tuning global. LR=1e-5 expertos, LR=1e-4 router. Early stopping.

El ablation study del router (Sección 4) se ejecuta en FASE 2 (Método B) o durante el warm-up (Método A), después de extraer los CLS tokens del backbone ViT preentrenado.




8. Optimización de Hardware

8.1 Técnicas Obligatorias
•	FP16 Mixed Precision: from torch.cuda.amp import autocast, GradScaler
•	Gradient Accumulation: ACCUMULATION_STEPS=4, batch efectivo = 32
•	Gradient Checkpointing: obligatorio para Expertos 4 y 5 (3D)
•	DataParallel: nn.DataParallel(model) para usar ambas tarjetas

8.2 Los Routers Estadísticos No Requieren GPU
GMM, Naive Bayes y k-NN se ejecutan completamente en CPU con sklearn y FAISS. El único componente GPU en el ablation es el backbone ViT para extraer los CLS tokens. Esto es una ventaja: el ablation study completo puede ejecutarse en CPU una vez que los embeddings están guardados en disco.

9. Cronograma de 4 Semanas

Semana	Fase	Actividades clave	Entregable
S9	Datos, preprocesado y expertos individuales	• Descarga y preprocesamiento de los 5 datasets • Implementación del AdaptivePreprocessor (2D/3D) • Entrenamiento individual de Expertos 1, 2, 3 (2D) • Elección y justificación de arquitecturas de expertos • Configuración del entorno multi-GPU	Checkpoint de cada experto 2D AdaptivePreprocessor validado Justificación de arquitecturas en borrador del reporte
S10	Expertos 3D, backbone ViT y ablation study	• Entrenamiento Expertos 4 y 5 (3D) con grad. checkpointing • Implementación backbone ViT para router • Extracción de CLS tokens de todo el dataset • Implementación y comparación de los 4 mecanismos de routing • Medición de Routing Accuracy por método • Implementación de Auxiliary Loss	Expertos 3D entrenados Tabla ablation study completa Router ganador seleccionado Auxiliary Loss implementada
S11	Sistema MoE completo y calibración	• Ensamble MoE con el router ganador del ablation • Entrenamiento completo del sistema MoE • Calibración de α (balanceo de carga) • Evaluación en validación: F1 Macro por dataset • OOD Detection con imágenes no médicas • Borrador del reporte técnico (secciones 1–3)	Sistema MoE funcionando Métricas de validación documentadas Cociente max/min(f_i) < 1.30 Borrador de reporte avanzado
S12	Dashboard, reporte y entrega final	• Dashboard Streamlit/Gradio con todas las funciones • Attention Heatmap del router ViT • Panel de Load Balance y OOD detection • Finalización del reporte técnico (7 páginas máx.) • Revisión de reproducibilidad (seeds, requirements.txt) • Demo en vivo ante el profesor	Dashboard funcional en localhost Reporte técnico PDF final (≤7 páginas) Repositorio Git documentado Demo con imagen desconocida

 
10. Métricas de Evaluación

Métrica	Umbral	¿Por qué es alcanzable?	¿Por qué es difícil?
F1-Score Macro (Datasets 2D)	> 0.72	EfficientNet/ConvNeXt con fine-tuning en ImageNet alcanzan ~0.70–0.78 en estos datasets.	Distribución muy desbalanceada en NIH (14 patologías). Requiere class-weighted loss y augmentation agresivo.
F1-Score Macro (Datasets 3D)	> 0.65	Modelos 3D livianos (R3D-18, Swin3D-Tiny) con grad. checkpointing y FP16 alcanzan ~0.65–0.72.	Alta variabilidad volumétrica con datos limitados (281 vol. de páncreas).
Routing Accuracy (ablation study)	> 0.80 para el mejor router	Un backbone ViT-Tiny preentrenado produce embeddings suficientemente discriminativos para >80% de routing correcto.	El router debe generalizar a volúmenes 3D aunque el backbone esté preentrenado en 2D.
OOD Detection (AUROC Router)	> 0.80	Entropía del gating score como señal de incertidumbre — no requiere entrenamiento adicional.	El router debe generalizar su concepto de imagen médica a tipos no vistos.
VRAM por tarjeta	< 11.5 GB	FP16 + gradient accumulation + grad. checkpointing (3D). Los routers estadísticos corren en CPU.	Router ViT + 2 expertos 3D simultáneos puede superar 12 GB sin optimizaciones.
Load Balance max(f_i)/min(f_i)	< 1.30 (solo ViT router)	Con α=0.01–0.05 y DataLoader proporcional, el balance se logra en 20+ épocas.	El desbalance natural (112K imgs 2D vs 281 vol. 3D) sesga el router si no se muestrea proporcionalmente.

 
11. Dashboard Interactivo

11.1 Funcionalidades Obligatorias
15.	Carga de imagen: PNG, JPEG, NIfTI. Detección automática 2D/3D.
16.	Preprocesado transparente: mostrar dimensiones originales y adaptadas.
17.	Inferencia en tiempo real: etiqueta, confianza, tiempo en milisegundos.
18.	Attention Heatmap del Router ViT: mapa de calor sobre la imagen original.
19.	Panel del experto activado: nombre, arquitectura, dataset de origen, gating score.
20.	Panel del ablation study: tabla comparativa de Routing Accuracy de los 4 métodos.
21.	Load Balance: gráfica de barras con f_i acumulado desde que arrancó el sistema.
22.	OOD Detection: alerta cuando la entropía del gating score supera el umbral calibrado.

 
12. Reporte Técnico: Estructura y Requisitos

Entregable obligatorio: Reporte técnico de máximo 7 páginas
El reporte es un entregable independiente del dashboard y el código.
Debe tener formato de artículo técnico breve: conciso, preciso, con tablas y figuras que soporten cada afirmación.
Fuente: Arial o Times New Roman 11pt. Interlineado 1.15. Márgenes 2.5 cm.
Extensión: mínimo 5 páginas, máximo 7 páginas (sin contar referencias ni portada).
Formato de entrega: PDF. Nombre del archivo: ReporteMoE_[NombreEquipo].pdf

12.1 Estructura Obligatoria del Reporte
Sección	Páginas	Contenido mínimo requerido
1. Introducción	0.5 pág.	Motivación del problema. Pregunta científica central. Contribución del equipo. No repetir el enunciado del proyecto.
2. Arquitectura Propuesta	1 pág.	Diagrama del sistema MoE implementado. Descripción del preprocesador adaptativo. Justificación de las arquitecturas de expertos elegidas (¿por qué esa y no otra?).
3. Ablation Study del Router	1.5 pág.	Tabla comparativa de los 4 mecanismos (Routing Accuracy, latencia, VRAM). Gráfica de evolución durante entrenamiento para ViT+Linear. Discusión: ¿qué método ganó y por qué? ¿Coincide con la hipótesis inicial?
4. Resultados de Clasificación	1.5 pág.	F1 Macro por dataset (tabla). Curvas de entrenamiento. Matriz de confusión del experto con menor F1. Comparación con baselines de la literatura.
5. Balance de Carga	0.5 pág.	Evolución de f_i durante el entrenamiento. Valor final de max(f_i)/min(f_i). α utilizado y proceso de calibración.
6. Alcances y Limitaciones	1 pág.	¿Qué funciona bien y por qué? ¿Qué no funcionó como se esperaba? Limitaciones de hardware. Limitaciones del dataset (imbalance, tamaño). ¿Qué haría el equipo con más tiempo/recursos?
7. Conclusiones	0.5 pág.	Respuesta directa a la pregunta científica. Lecciones aprendidas sobre MoE en imágenes médicas.
Referencias	Sin límite	Mínimo 5 referencias. Incluir los papers del router elegido, los datasets y Switch Transformer.

12.2 Figuras Obligatorias en el Reporte
•	Figura 1: Diagrama de arquitectura del sistema MoE implementado (puede ser el del dashboard).
•	Figura 2: Tabla comparativa del ablation study (los 4 routers con sus métricas).
•	Figura 3: Curvas de entrenamiento (loss y F1 Macro) del sistema MoE completo.
•	Figura 4: Evolución del balance de carga f_i durante el entrenamiento.
•	Figura 5: Attention Heatmap del router sobre al menos 3 imágenes de diferentes modalidades.


12.3 Lo que NO debe incluir el reporte
⚠  Errores comunes que reducen la calificación del reporte
• No transcribir secciones del enunciado del proyecto — el reporte reporta sus resultados, no repite la tarea.
• No incluir bloques de código fuente en el reporte — el código va en el repositorio Git.
• No afirmar resultados sin tabla o figura que los soporten.
• No omitir resultados negativos — si un método no funcionó, explicar por qué es más valioso que ocultarlo.
• No exceder 7 páginas — la capacidad de síntesis es parte de la evaluación.

 
13. Rúbrica de Evaluación

Componente	% Nota	Criterios
Ablation Study del Router (4 mecanismos comparados)	25%	Los 4 mecanismos implementados y funcionando. Tabla comparativa con Routing Accuracy. Discusión del resultado. Router ganador justificado.
Sistema MoE completo (router + expertos + auxiliary loss)	20%	Vision Transformer como router del sistema final. Expertos heterogéneos. Auxiliary Loss implementada desde cero. Preprocesador adaptativo funcional.
Métricas de clasificación (F1 Macro por dataset)	20%	2D: F1>0.72 (full), >0.65 (aceptable). 3D: F1>0.65 (full), >0.58 (aceptable). Routing Accuracy >0.80 para el mejor router.
Dashboard interactivo (demo en vivo)	15%	Todas las funciones obligatorias presentes. Panel del ablation study visible. Heatmap del router. OOD detection activo. Demo fluida.
Reporte técnico (máx. 7 páginas)	15%	Estructura completa. Pregunta científica respondida con datos. Figuras obligatorias presentes. Alcances y limitaciones honestos. Máx. 7 páginas.
Repositorio Git y reproducibilidad	5%	README con instrucciones. requirements.txt. Seeds fijas. El sistema corre en el clúster del curso sin modificaciones.
PENALIZACIÓN — Load Balance max(f_i)/min(f_i) > 1.30	-40%	Solo aplica al router ViT+Linear del sistema final. Descuenta 40% del puntaje total obtenido.
PENALIZACIÓN — Uso de metadatos externos	-20%	Si cualquier componente del sistema recibe etiqueta de modalidad, texto o metadato como entrada.

 
14. Referencias y Recursos

14.1 Papers Fundamentales
•	Fedus, W., Zoph, B., & Shazeer, N. (2021). Switch Transformers: Scaling to Trillion Parameter Models. JMLR. [Auxiliary Loss, Load Balancing]
•	Jacobs, R. A., Jordan, M. I., Nowlan, S. J., & Hinton, G. E. (1991). Adaptive Mixtures of Local Experts. Neural Computation. [Origen histórico del MoE — motivación del ablation]
•	Wan, K. et al. (1993). Multispectral image classification using a mixture density model. SPIE Vol. 1766. [Ancestro estadístico del router MoE]
•	Dosovitskiy, A. et al. (2020). An Image is Worth 16×16 Words. ICLR 2021. [ViT — backbone del router]
•	Wu, H. et al. (2021). CvT: Introducing Convolutions to Vision Transformers. ICCV 2021.
•	Liu, Z. et al. (2021). Swin Transformer: Hierarchical Vision Transformer. ICCV 2021.
•	Hihn, H. & Braun, D.A. (2024). Mixture of Experts for Image Classification: What's the Sweet Spot? arXiv:2411.18322. [Comparación de mecanismos de gating en visión]

14.2 Implementación
•	Curso MoE — ApX Machine Learning: https://apxml.com/courses/mixture-of-experts-advanced-implementation
•	HuggingFace Transformers MoE: https://github.com/huggingface/transformers/issues/40145
•	timm: https://github.com/huggingface/pytorch-image-models [ViT, CvT, EfficientNet, ConvNeXt preentrenados]
•	FAISS: https://github.com/facebookresearch/faiss [Indexación para k-NN router]
•	sklearn GaussianMixture: https://scikit-learn.org/stable/modules/mixture.html
•	MONAI: https://monai.io [Datasets médicos 3D]

14.3 Datasets
•	NIH ChestX-ray14: https://www.kaggle.com/datasets/nih-chest-xrays/data
•	ISIC 2019: https://www.kaggle.com/datasets/andrewmvd/isic-2019
•	Osteoarthritis Knee: https://www.kaggle.com/datasets/dhruvacube/osteoarthritis
•	LUNA16: https://www.kaggle.com/datasets/fanbyprinciple/luna-lung-cancer-dataset
•	Pancreatic Cancer: https://zenodo.org/records/13715870










Requisito de ajuste ABET
 






Documento elaborado para el curso Incorporar Elementos de IA — Unidad II, Bloque Visión: Clasificación. Hardware: 2 tarjetas gráficas de 12 GB de VRAM cada una. Período: Semanas 9–12.
