# Análisis de Código: Multi-Expert Vision Router (Paso a Paso)

Este documento explica los bloques de código reales clave de tu notebook (`05_Router_Profesor_Fase3_Solo.ipynb`). 

---

## Paso 1: El DataLoader Mixto (`MixedMedicalDataset`)

El desafío de la Fase 3 era manejar 5 datasets (3 de imágenes 2D y 2 de volúmenes 3D) en simultáneo, asegurando el balance para el Gating. 

```python
# Extracto: Creación de índices en build_dataset_label_index
def build_dataset_label_index(roots):
    idx = {'nih': {}, 'isic': {}, 'osteo': {}, 'luna16': {}, 'meta': {}}
    # Ejemplo con NIH... Lee Data_Entry_2017.csv y hace multi-label (14)
    # LUNA16/Páncreas indexan .npz o .mhd para volúmenes.
    return idx
```
**Importancia:** De aquí se alimenta el `WeightedRandomSampler` que garantiza que las frecuencias de *ISIC* o *Páncreas* no queden aplastadas por las gigantes de *NIH*, pasando al Router un Batch equilibrado.

---

## Paso 2: La Adaptación Dimensional (`SwitchablePatchEmbed`)

Poner `[3, 224, 224]` y `[3, 16, 128, 128]` en un mismo Transformer provocaría un `Dimension error`. La magia ocurre en la clase adaptativa:

```python
class SwitchablePatchEmbed(nn.Module):
    def __init__(self, embed_dim=192, patch_size_2d=16, patch_size_3d=8, in_channels_2d=3):
        super().__init__()
        self.embed_dim = embed_dim

        # D: Patch Embedding 2D (Conv2d 16x16)
        self.patch_embed_2d = PatchEmbed(
            spatial_dims=2, in_chans=in_channels_2d, patch_size=patch_size_2d, embed_dim=embed_dim
        )
        # E: Patch Embedding 3D (Conv3d 8x8x8)
        self.patch_embed_3d = PatchEmbed(
            spatial_dims=3, in_chans=1, patch_size=patch_size_3d, embed_dim=embed_dim
        )
        # Token CLS
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
```
**Importancia:** Esta clase intercepta cada ítem del `batch`, lee su dimensión y le aplica la convolución respectiva. El resultado para TODOS los ítems es una serie plana de vectores de dimensión 192 (tokens).

---

## Paso 3: El Gating del Router (`VisionRouter`)

La cabeza linear final calcula en qué porcentaje confía el Router para cada panel experto:

```python
class VisionRouter(nn.Module):
    def __init__(self, ...):
      ...
      self.router_head = nn.Linear(embed_dim, num_experts) # De 192 -> 5 neuronas

    def forward(self, x):
        tokens = self.patch_embed(x) 
        out = self.vit(tokens) 
        cls_feat = out[:, 0, :]   # Token CLS global
        logits = self.router_head(cls_feat)
        return logits             # Salida a Softmax
```

---

## Paso 4: Evitando el Colapso Matemáticamente (L_aux)

Aquí está la implementación del Aux Loss (Switch Transformer):

```python
def _switch_aux_loss(router_probs: torch.Tensor, num_experts: int) -> torch.Tensor:
    B = router_probs.size(0)
    # Frecuencia dura 
    hard = router_probs.argmax(dim=1)
    with torch.no_grad():
        f = torch.bincount(hard, minlength=num_experts).float() / float(B)
    
    # Promedio suave de probabilidades
    P = router_probs.mean(dim=0)
    # Penalidad multiplicativa
    return num_experts * (f * P).sum()
```
**Importancia:** Obliga al modelo a esparcir las cargas y evitar que todos los píxeles deriven al mismo experto (ratio < 1.30).

---

## Paso 5: La Función de Coste Condicionada (Teacher-Student)

Extraído de `fit_router_with_eval`.

```python
          # ¡PUNTO CLAVE! Inferencia aisada de experto.
          expert_out = expert_model(batch_x[i:i+1])
          loss_experto = expert_criteria(expert_out, verdadera_etiqueta[i:i+1])
          
          losses_task.append(loss_experto)
      
      L_task_total = torch.stack(losses_task).mean()

  # Pérdida final acumulada
  loss = L_routing + L_task_total + (ALPHA_AUX * L_aux)
  loss.backward()
```
**Importancia:** En lugar de procesar el batch gignate (`N=64`) en los 5 modelos densos (multiplicando tiempos), **sólo procesa 64 inferencias**. Ahorrando valiosos gigabytes de VRAM.
