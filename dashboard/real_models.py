"""
real_models.py — Arquitecturas reales del sistema MoE.

Extraídas de 05_Router_Profesor_Fase3_Solo.ipynb.
Incluye:
  - VisionRouter (SwitchablePatchEmbed + ViT-Tiny + Linear head)
  - LungMaxViT (Experto 1 — NIH)
  - build_efficientnet_b3_expert() (Experto 2 — ISIC)
  - build_vgg16_bn_expert() (Experto 3 — Osteoarthritis)
  - DCSwinBStyle3D (Experto 4 — LUNA16)
  - R3D18Expert (Experto 5 — Pancreas)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import torchvision.models as models
from torchvision.models.video import r3d_18
from torch.utils.checkpoint import checkpoint as grad_checkpoint

# Importar PatchEmbed de MONAI (necesario para SwitchablePatchEmbed)
try:
    from monai.networks.blocks import PatchEmbed
    HAS_MONAI = True
except ImportError:
    HAS_MONAI = False


# =============================================================================
# 1. Componentes del Router (VisionRouter)
# =============================================================================

class SwitchablePatchEmbed(nn.Module):
    """
    Switchable Patch Embedding para manejar entradas 2D y 3D.
    Usa MONAI PatchEmbed internamente.
    """

    def __init__(self, embed_dim=192, patch_size_2d=16, patch_size_3d=8,
                 in_channels_2d=3):
        super().__init__()
        if not HAS_MONAI:
            raise ImportError(
                "MONAI es necesario para SwitchablePatchEmbed. "
                "Instalar con: pip install monai"
            )
        self.embed_dim = embed_dim
        self.patch_embed_2d = PatchEmbed(
            spatial_dims=2,
            in_chans=in_channels_2d,
            patch_size=patch_size_2d,
            embed_dim=embed_dim,
        )
        self.patch_embed_3d = PatchEmbed(
            spatial_dims=3,
            in_chans=1,
            patch_size=patch_size_3d,
            embed_dim=embed_dim,
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, 513, embed_dim))

    def _patch_tokens_to_sequence(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """Convierte la salida de PatchEmbed a una secuencia [T, embed_dim]."""
        x = patch_tokens
        if x.dim() == 3 and x.size(-1) == self.embed_dim:
            return x.squeeze(0) if x.size(0) == 1 else x.reshape(
                -1, self.embed_dim)
        if x.dim() >= 3 and x.size(-1) == self.embed_dim:
            return x.reshape(-1, self.embed_dim)
        if x.dim() == 4 and x.size(1) == self.embed_dim:
            x = x.flatten(2).transpose(1, 2).contiguous()
            return x.reshape(-1, self.embed_dim)
        if x.dim() == 5 and x.size(1) == self.embed_dim:
            x = x.flatten(2).transpose(1, 2).contiguous()
            return x.reshape(-1, self.embed_dim)
        return x

    def forward(self, batch_tensors):
        """
        batch_tensors: un tensor o lista de tensores.
        En el dashboard procesamos 1 sample a la vez.
        """
        if not isinstance(batch_tensors, (list, tuple)):
            batch_tensors = [batch_tensors]

        batch_size = len(batch_tensors)
        tokens_list = []

        for sample in batch_tensors:
            if sample.ndim == 3:  # [C, H, W] -> [1, C, H, W]
                sample = sample.unsqueeze(0)

            if sample.ndim == 4:  # 2D image
                if sample.shape[1] == 1:
                    sample = sample.repeat(1, 3, 1, 1)
                patch_tokens = self.patch_embed_2d(sample)
            elif sample.ndim == 5:  # 3D volume
                patch_tokens = self.patch_embed_3d(sample)
            else:
                raise ValueError(f"Dimensión de entrada inválida: {sample.ndim}")

            seq = self._patch_tokens_to_sequence(patch_tokens)
            tokens_list.append(seq)

        padded = torch.nn.utils.rnn.pad_sequence(
            tokens_list, batch_first=True)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, padded), dim=1)
        x = x + self.pos_embed[:, :x.size(1), :]
        return x


class VisionRouter(nn.Module):
    """
    VisionRouter basado en ViT-Tiny (vit_tiny_patch16_224 de timm).
    Incluye un hook para capturar los attention weights del último bloque
    (necesario para generar el Attention Heatmap, consigna #18).
    """

    def __init__(self, embed_dim=192, num_experts=5, pretrained=False,
                 **kwargs):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_experts = num_experts
        self.patch_embed = SwitchablePatchEmbed(embed_dim=embed_dim)

        self.vit = timm.create_model(
            'vit_tiny_patch16_224',
            pretrained=pretrained,
            num_classes=0,
            global_pool='',
        )
        # Reemplazar el patch embedding del ViT por Identity
        # porque usamos SwitchablePatchEmbed
        self.vit.patch_embed = nn.Identity()

        self.router_head = nn.Linear(embed_dim, num_experts)

        # Almacena el mapa de atención del último bloque
        self._last_attn_weights = None
        self._hook_handle = None
        self._register_attn_hook()

    def _register_attn_hook(self):
        """Registra un forward hook en el módulo de atención del último
        bloque del ViT para capturar los attention weights."""
        try:
            last_block = self.vit.blocks[-1]
            attn_module = last_block.attn

            def hook_fn(module, input, output):
                # En timm, el módulo Attention guarda attn_weights
                # internamente si configuramos attn_drop correctamente.
                # Usamos un enfoque alternativo: recalcular la atención.
                pass

            # Inyectamos un forward hook que captura qkv
            self._hook_handle = attn_module.register_forward_hook(
                self._attn_hook
            )
        except (AttributeError, IndexError):
            pass  # Si la estructura del ViT no lo permite, pasamos

    def _attn_hook(self, module, input, output):
        """Hook que captura los attention weights del módulo de atención."""
        # En timm ViT, la atención se computa internamente.
        # Necesitamos recalcularla a partir de qkv.
        try:
            x = input[0]
            B, N, C = x.shape
            qkv = module.qkv(x).reshape(
                B, N, 3, module.num_heads, C // module.num_heads
            ).permute(2, 0, 3, 1, 4)
            q, k, _ = qkv.unbind(0)
            head_dim = C // module.num_heads
            attn = (q @ k.transpose(-2, -1)) * (head_dim ** -0.5)
            attn = attn.softmax(dim=-1)
            self._last_attn_weights = attn.detach()
        except Exception:
            self._last_attn_weights = None

    def forward(self, batch_tensors):
        """
        Args:
            batch_tensors: tensor o lista de tensores preprocesados

        Returns:
            logits:       [B, num_experts] — gating logits
            cls_token:    [B, embed_dim] — CLS token embedding
            attn_weights: [B, num_heads, seq_len, seq_len] o None
        """
        x = self.patch_embed(batch_tensors)

        # Pasar por los bloques del ViT
        # (el hook capturará la atención del último bloque)
        for blk in self.vit.blocks:
            x = blk(x)

        # Aplicar LayerNorm final del ViT (vit.norm)
        # Esto es CRÍTICO: el checkpoint incluye vit.norm.weight y vit.norm.bias
        # entrenados. Sin esta normalización, el CLS token tiene una distribución
        # completamente diferente y el router_head produce predicciones incorrectas.
        x = self.vit.norm(x)

        cls_token = x[:, 0]
        logits = self.router_head(cls_token)

        return logits, cls_token, self._last_attn_weights


# =============================================================================
# 2. Expertos Reales
# =============================================================================

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for LungMaxViT initial state."""
    def __init__(self, in_channels, reduced_channels):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, in_channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _, _ = x.size()
        y = x.view(b, c, -1).mean(dim=2)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class LungMaxViT(nn.Module):
    """Experto 1 — NIH ChestX-ray14. LungMaxViT (MaxViT modificado)."""

    def __init__(self, num_classes=14):
        super().__init__()
        self.initial_block = nn.ModuleDict({
            'conv1': nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.GELU()
            ),
            'conv2': nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(64),
                nn.GELU()
            ),
            'conv3': nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.GELU()
            ),
            'se': SEBlock(64, 16),
            'proj': nn.Sequential(
                nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
                nn.BatchNorm2d(3)
            )
        })
        
        self.maxvit = timm.create_model(
            'maxvit_tiny_tf_224',
            pretrained=False,
            num_classes=num_classes,
        )

    def forward(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(0)
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
            
        t = self.initial_block['conv1'](x)
        t = self.initial_block['conv2'](t)
        t = self.initial_block['conv3'](t)
        t = self.initial_block['se'](t)
        t = self.initial_block['proj'](t)
        
        return self.maxvit(t)


def build_efficientnet_b3_expert(num_classes=9):
    """Experto 2 — ISIC 2019. EfficientNet-B3."""
    model = timm.create_model(
        "efficientnet_b3",
        pretrained=False,
        num_classes=num_classes,
    )
    return model


def build_vgg16_bn_expert(num_classes=5):
    """
    Experto 3 — Osteoarthritis. VGG-16 BN con:
    - Entrada de 1 canal (radiografía)
    - Clasificador custom con BatchNorm y Dropout
    """
    model = models.vgg16_bn(weights=None)
    # Adaptar primera conv para 1 canal
    model.features[0] = nn.Conv2d(1, 64, kernel_size=3, padding=1)
    # Clasificador custom (del notebook)
    model.classifier = nn.Sequential(
        nn.Linear(512 * 7 * 7, 512),
        nn.ReLU(True),
        nn.BatchNorm1d(512),
        nn.Dropout(0.5),
        nn.Linear(512, 256),
        nn.ReLU(True),
        nn.BatchNorm1d(256),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes),
    )
    return model


class DCSwinBStyle3D(nn.Module):
    """
    Experto 4 — LUNA16. Dual-branch (Swin2D adaptado a 3D + CNN 3D).
    Usa gradient checkpointing obligatorio (consigna).
    """

    def __init__(self, num_classes=2):
        super().__init__()
        # Rama Swin: tomar componentes de swin_tiny_2D
        swin2d = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=False,
            num_classes=0,
            img_size=256,
        )
        c_out = swin2d.patch_embed.proj.out_channels
        self.patch_embed_3d = nn.Conv3d(
            1, c_out, kernel_size=(4, 4, 4), stride=(4, 4, 4)
        )
        self.dw_mix = nn.Conv3d(
            c_out, c_out, kernel_size=3, padding=1, groups=c_out
        )
        self.patch_embed_norm = swin2d.patch_embed.norm
        self.layers = swin2d.layers
        self.norm = swin2d.norm
        self.swin_dim = swin2d.num_features

        # Rama CNN 3D: basada en R3D-18
        base = r3d_18(weights=None)
        self.cnn_branch = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2),
                      padding=(1, 3, 3), bias=False),
            base.stem[1],
            base.stem[2],
            base.layer1,
            base.layer2,
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
        )
        self.cnn_dim = 128

        # Head de fusión
        self.head = nn.Linear(self.swin_dim + self.cnn_dim, num_classes)

    def forward(self, x):
        if x.ndim == 4:
            x = x.unsqueeze(0)

        # Rama CNN
        z_cnn = self.cnn_branch(x)

        # Rama Swin adaptada a 3D
        t = self.patch_embed_3d(x)
        t = self.dw_mix(t)
        b, c, d, h, w = t.shape
        # Aplanar las dimensiones 3D a 2D para los layers de Swin
        t = t.view(b, c, 4, 4, h, w) \
             .permute(0, 1, 2, 4, 3, 5) \
             .reshape(b, c, 4 * h, 4 * w) \
             .permute(0, 2, 3, 1)
        t = self.patch_embed_norm(t)

        # Gradient checkpointing obligatorio para 3D
        for layer in self.layers:
            t = grad_checkpoint(layer, t, use_reentrant=False)

        t = self.norm(t)
        z_swin = t.mean(dim=[1, 2])

        return self.head(torch.cat([z_swin, z_cnn], dim=1))


class R3D18Expert(nn.Module):
    """
    Experto 5 — Pancreas. R3D-18 adaptado para 1 canal.
    Usa gradient checkpointing obligatorio (consigna).
    Estructura ajustada para coincidir con las keys del state_dict real.
    """

    def __init__(self, num_classes=2):
        super().__init__()
        base = r3d_18(weights=None)
        
        # El checkpoint fue entrenado con 3 canales de entrada (no 1),
        # así que NO modificamos stem[0]; se mantiene la conv original [64, 3, 3, 7, 7].
        
        # Atributos individuales para coincidir con keys del state_dict (sin prefijo backbone)
        self.stem = base.stem
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = base.avgpool
        
        # Head reconstruida: el checkpoint usa 128 unidades ocultas (no 256)
        self.head = nn.Sequential(
            nn.Flatten(1),                  # 0
            nn.Linear(512, 128),           # 1 (weight, bias)
            nn.BatchNorm1d(128),           # 2 (weight, bias, running_mean...)
            nn.ReLU(True),                 # 3
            nn.Dropout(0.4),               # 4
            nn.Linear(128, num_classes)    # 5 (weight, bias)
        )

    def forward(self, x):
        if x.ndim == 4:
            x = x.unsqueeze(0)

        # Gradient checkpointing obligatorio por bloque para optimizar memoria en 3D
        x = grad_checkpoint(self.stem, x, use_reentrant=False)
        x = grad_checkpoint(self.layer1, x, use_reentrant=False)
        x = grad_checkpoint(self.layer2, x, use_reentrant=False)
        x = grad_checkpoint(self.layer3, x, use_reentrant=False)
        x = grad_checkpoint(self.layer4, x, use_reentrant=False)

        x = self.avgpool(x)
        x = self.head(x)
        
        return x
