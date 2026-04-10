# Resumen: M3ViT — Mixture-of-Experts ViT para Multi-task Learning Eficiente
## *Liang, Fan, Sarkar et al. — UT Austin, Georgia Tech, Texas A&M. NeurIPS 2022*

**Fuente:** `guides/Moes Medical Vit/M3ViT.md`  
**Repositorio:** https://github.com/VITA-Group/M3ViT

---

## 1. De que trata el paper en 3 lineas

M3ViT propone reemplazar las capas MLP densas dentro de un Vision Transformer por capas MoE (grupos de expertos MLP), donde un router por tarea elige cuales expertos activar. Esto resuelve dos problemas clasicos del Multi-task Learning: los conflictos de gradientes entre tareas durante el entrenamiento, y la ineficiencia de activar el modelo completo para ejecutar una sola tarea en inferencia. El resultado es un sistema que logra mejor accuracy multiarea que los baselines y reduce 88% de los FLOPs en inferencia por tarea.

---

## 2. Arquitectura M3ViT

El cambio central es simple: reemplazar las capas FFN (MLP) del ViT por capas MoE.

**ViT estandar:**
```
Input → Patch Embedding → Self-Attention → MLP (denso) → Output
```

**M3ViT:**
```
Input → Patch Embedding → Self-Attention → MoE Layer → Output
                                               |
                                    Router por tarea
                                    selecciona K de N expertos MLP
```

### Formula del Router (Top-K Gating)

$$y = \sum_{k=1}^{K} R(x)_k \cdot f_k(x)$$

$$R(x) = \text{TopK}(\text{softmax}(G(x)), K)$$

Donde:
- $G(x)$: red MLP single-layer que calcula logits por experto
- $K = 4$ expertos activos de $N = 16$ candidatos totales
- Cada experto: $f_k(x) = W_2 \cdot \sigma_{GELU}(W_1 x)$

Los expertos tienen un cuarto del tamano de los MLPs originales del ViT para mantener los FLOPs equivalentes.

### Dos variantes de routing para MTL

**Multi-gate MoE (la que da mejor resultado):**  
Cada tarea tiene su propio router $R_i$. Los expertos son compartidos entre tareas pero cada tarea elige los suyos de forma independiente.

$$y_i = \sum_{k=1}^{K} R_i(x)_k \cdot f_k(x)$$

**Task-conditioned MoE (variante alternativa):**  
Un solo router compartido recibe como input el token concatenado con un embedding de tarea (one-hot procesado por MLP):

$$y_i = \sum_{k=1}^{K} R(x, t_i)_k \cdot f_k(x), \quad t_i = \text{ReLU}(T(x, e_i))$$

---

## 3. Resultados Principales

**PASCAL-Context (5 tareas visuales):**

| Modelo | Backbone | Delta_m (%) | FLOPs (G) |
|---|---|---|---|
| Cross-Stitch | ResNet-18 | +0.60 | 647 |
| M-ViT (MTL base) | ViT-small | -1.77 | 83 |
| M2ViT (+MoE) | MoE ViT-small | **+2.71** | **84** |
| M3ViT (+hardware) | MoE ViT-small | **+2.71** | **84** |

M2ViT supera al mejor metodo previo (+0.60) y reduce los FLOPs de Cross-Stitch en 88% (de 647G a 84G).

**Hallazgo clave del paper:** La ganancia de accuracy del MoE aumenta a medida que se agregan mas tareas al sistema. Con 2 tareas la mejora es moderada; con 5 tareas es significativa y supera todos los baselines.

---

## 4. Balanceo de Carga

El paper usa la misma loss de balanceo que Switch Transformers con peso 0.01:

> *"We also employ the load and important balancing loss with the weight of 0.01 following [30] to avoid always picking the same experts while ignoring others."*

Esto confirma que el valor $\alpha = 0.01$ es el estandar de la literatura para la Auxiliary Loss de tu proyecto.

---

## 5. Diferencias con tu proyecto

| Aspecto | M3ViT (Paper) | Tu Proyecto (Consigna) |
|---|---|---|
| Dominio | Vision natural (segmentacion, depth, bordes) | Imagenes medicas multimodales |
| Expertos | MLPs identicos dentro del ViT (micro) | Redes heterogeneas por modalidad (macro) |
| Router | Top-K por tarea, condicionado a label de tarea | Sin labels de tarea, solo la imagen |
| Nivel del MoE | Dentro de cada capa Transformer | Entre el backbone y los expertos externos |
| Escala | 16 expertos MLP pequenos | 5 expertos redes completas |
| Hardware | Co-diseno FPGA | Google Colab GPU |

La diferencia principal es la misma que con MedMoE: M3ViT condicionado por el label de tarea, tu proyecto prohibe usar esa informacion.

---

## 6. Lo que si puedes usar para tu proyecto

### Para el marco teorico del reporte

El paper demuestra formalmente que el MoE aplicado a ViT resuelve el problema de conflictos de gradientes entre dominios distintos. En tu caso el problema analogo es: si entrenas una sola red con las 5 modalidades, los gradientes de radiografias de torax (100K muestras) van a dominar y borrar los gradientes del pancreas (557 muestras). El argumento del paper que justifica tu arquitectura MoE es:

> *"negative cosine similarities between different tasks' gradients are detrimental [...] conflicting gradients not only slow down convergence but also bias the learned representations against some tasks"*

### Para citar la Auxiliary Loss

El paper usa $\alpha = 0.01$ para la balancing loss. Esto es independiente del dominio y confirma el valor recomendado por Switch Transformers. Puedes citar tanto Switch Transformers como M3ViT para respaldar tu eleccion de $\alpha$.

### Para el argumento de eficiencia computacional

M3ViT demuestra que activar solo los expertos relevantes en inferencia reduce FLOPs en 88% sin perder accuracy. En tu proyecto, en inferencia solo activa 1 de 5 expertos, lo que representa una reduccion del 80% del computo de los expertos. Cita este resultado para justificar la decision de arquitectura en el reporte.

### Para el ablation study del routing

La Tabla 3 del paper compara tres tipos de router (single, multi-gate, task-conditioned). El resultado es que multi-gate supera a task-conditioned por un margen consistente. Esto es una referencia para justificar por que comparar multiples mecanismos de routing (tu Ablation Study) es una practica estandar en la literatura.

---

## 7. Como citar en el reporte tecnico (IEEE)

```
[X] H. Liang, Z. Fan, R. Sarkar, Z. Jiang, T. Chen, K. Zou, Y. Cheng, C. Hao,
    and Z. Wang, "Mixture-of-Experts Vision Transformer for Efficient Multi-task
    Learning with Model-Accelerator Co-design," in Proc. NeurIPS, 2022.
    arXiv:2210.14793.
```

---

## 8. Resumen de relevancia para la consigna

| Uso en el proyecto | Importancia |
|---|---|
| Justificar que MoE en ViT resuelve conflictos de gradientes entre dominios | Alta |
| Respaldar el valor alpha = 0.01 para la Auxiliary Loss | Alta |
| Marco teorico del Ablation Study de mecanismos de routing | Media |
| Justificar eficiencia del sparse activation (solo 1 experto activo en inferencia) | Media |
| Replicar su arquitectura exacta (MLP experts dentro del ViT) | No aplica — diferente nivel de abstraccion |
