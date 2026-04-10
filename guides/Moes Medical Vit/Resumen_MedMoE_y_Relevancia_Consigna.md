# 📄 Resumen: MedMoE — Modality-Specialized Mixture of Experts
## *Chopra, Mao, Sanchez-Rodriguez et al. — Georgia Tech + Emory, 2025*

> **Fuente:** `guides/Modality-Specialized Mixture of Experts for Medical Vision-Language Understanding.md`

---

## 1. ¿De qué trata el paper en 3 líneas?

MedMoE propone usar un **Mixture of Experts dentro del extractor de features visual** de un modelo médico imagen-texto. La idea central es que las imágenes médicas de diferentes modalidades (X-ray, CT, MRI, ultrasonido) necesitan **estrategias de extracción de features distintas**: los X-rays requieren atención a patrones globales, mientras que CT/MRI requieren foco en lesiones locales pequeñas. Un solo encoder compartido para todo es insuficiente.

---

## 2. Arquitectura del MedMoE

```
Imagen Médica
      │
      ▼
┌─────────────────────────────┐
│  Swin Transformer Backbone  │  ← Extrae features multi-escala
│  F(1), F(2), F(3), F(4)    │  ← 4 resoluciones espaciales
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  MoE Router (MLP liviano)   │  ← Condicionado al global embedding
│  α = softmax(W2·ReLU(W1·t)) │  ← Elige experto k* = argmax(α)
└─────────────┬───────────────┘
              │
      ┌───────┴──────────┐
      ▼                  ▼
  Experto k*          Expertos 1..K
 (activo)            (inactivos en inferencia)
      │
      ▼
  V_local = Σ β(l) · F(l)  ← Combinación multi-escala adaptativa
      │
      ▼
Loss Global + Loss Local + λ·Loss Aux (CrossEntropy tipo report)
```

### Fórmulas clave

**Selección del experto:**
$$\boldsymbol{\alpha} = \text{softmax}(W_2 \cdot \text{ReLU}(W_1 \cdot \mathbf{t}_g)), \quad k^* = \arg\max_k \alpha_k$$

**Representación local del experto k:**
$$\mathbf{V}_k = \sum_{\ell=1}^{L} \beta_k^{(\ell)} \cdot \mathbf{F}_k^{(\ell)}$$

**Loss total:**
$$\mathcal{L}_{total} = \mathcal{L}_{global} + \mathcal{L}_{local} + \lambda \cdot \mathcal{L}_{aux}$$

---

## 3. Resultados Principales

| Dataset | Modalidad | Mejor Baseline | MedMoE | Mejora |
|---|---|---|---|---|
| RSNA | X-ray | 65.90% (UniMed) | **66.03%** | +0.13% |
| Breast Ultrasound | US | 74.99% (PMC-CLIP) | **78.32%** | +3.33% |
| ACL | MRI | 85.82% (UniMed) | 77.92% | −7.9% |
| CT Axial | CT | 39.97% (UniMed) | **40.00%** | +0.03% |

*Gana en 6/9 benchmarks en zero-shot*

**Eficiencia:** Solo activa 1 experto en inferencia → FLOPs similares a MedCLIP (Swin-Tiny base), con 37M params totales pero solo 7.8G FLOPs activos.

---

## 4. Diferencias clave entre MedMoE y tu proyecto

| Aspecto | **MedMoE (Paper)** | **Tu Proyecto (Consigna)** |
|---|---|---|
| **¿Qué enruta?** | Features locales dentro del extractor visual | La imagen completa (a qué experto enviarla) |
| **¿Cuándo decide?** | Dentro del backbone (nivel microscopico) | Antes del backbone (nivel macro) |
| **Input del Router** | Embedding del reporte de texto (`t_g`) | Solo la imagen, sin metadatos |
| **Backbone** | Swin Transformer | ViT (vit_base_patch16_224 de timm) |
| **Nivel de abstracción** | Routing de features multi-escala | Routing de pacientes completos a doctores |
| **¿Sin metadatos?** | NO — usa el reporte como input del router | **SÍ — prohibido usar metadatos (-20% nota)** |

> ⚠️ **Diferencia crítica:** MedMoE usa el reporte clínico de texto para condicionar el routing. Tu consigna **penaliza con -20%** si pasas cualquier metadato al router. Tu router solo puede ver la imagen/volumen.

---

## 5. ¿Qué SÍ puedes tomar de este paper para tu proyecto?

### ✅ 1. Marco teórico del Reporte (Sección 1 y 2)
La **motivación** del paper es exactamente tu argumento. Puedes citar:
> *"CT/MRI reports often emphasize local abnormalities, while interpretations of X-rays often focus on global patterns"*

Esto justifica por qué necesitas 5 expertos heterogéneos en lugar de 1 red monolítica.

### ✅ 2. La Auxiliary Loss como concepto compartido
MedMoE usa:
$$\mathcal{L}_{aux} = \text{CrossEntropy}(W_c \cdot \mathbf{t}_g, y)$$
donde $y$ es el tipo de reporte (CT, X-ray, MRI).

Tu consigna te pide la versión de Switch Transformer:
$$L_{aux} = \alpha \cdot N \cdot \sum f_i \cdot P_i$$

**Ambos buscan lo mismo:** evitar que el router ignore ciertos expertos. Son dos implementaciones del mismo concepto. ¡Cita ambas en tu reporte!

### ✅ 3. Dato de eficiencia computacional (Tabla 3 del paper)
Puedes usar esto en tu reporte para justificar el uso de Hard Routing (Top-1) en tu proyecto:

> *"MedMoE usa selección hard de expertos en inferencia, activando solo uno a la vez, manteniendo FLOPs comparables a la línea base monolítica (7.8G vs 4.5G de MedCLIP)"*

### ✅ 4. Validación empírica de que el MoE médico funciona
Los resultados de MedMoE (best en 6/9 benchmarks) son evidencia académica de que la **especialización modal funciona en imágenes médicas**. Cítalo como "evidencia de que nuestro enfoque MoE heterogéneo está respaldado por literatura reciente".

### ✅ 5. Concepto de Granularidad Diagnóstica por Modalidad
Este es el argumento más potente del paper. Para tu reporte, puedes argumentar:
- X-rays (NIH): necesitan atención a **patrones globales** (área pulmonar completa)
- CT 3D (LUNA/Páncreas): necesitan atención a **lesiones locales** (nódulos de 3-5mm)
- Dermatología (ISIC): necesitan atención a **texturas finas** y bordes de lesión
- Ortopedia (Knee): necesitan atención a **espacio articular y osteofitos**

Esto **justifica académicamente** por qué usas 5 expertos distintos y no uno solo.

---

## 6. Cómo citar en tu Reporte Técnico (IEEE/ABET)

```
[X] S. Chopra, L. Mao, G. Sanchez-Rodriguez, A. J. Feola, J. Li, and Z. Kira,
    "MedMoE: Modality-Specialized Mixture of Experts for Medical Vision-Language
    Understanding," Georgia Institute of Technology, 2025.
```

También puedes citar la frase del abstract directamente:

> *"Different medical imaging modalities capture diagnostic information at varying spatial resolutions, from coarse global patterns to fine-grained localized structures"* — MedMoE (Chopra et al., 2025)

---

## 7. Resumen Ejecutivo de Relevancia

| ¿Para qué sirve en tu proyecto? | Calificación |
|---|---|
| Citar como inspiración de la arquitectura MoE médica | ⭐⭐⭐ Esencial |
| Justificar los 5 expertos heterogéneos | ⭐⭐⭐ Esencial |
| Validar que Hard Routing (Top-1) es eficiente | ⭐⭐ Importante |
| Replicar su arquitectura exacta en tu código | ❌ No hacer — viola la consigna (usa el reporte de texto) |
| Marco teórico del reporte técnico | ⭐⭐⭐ Esencial |
