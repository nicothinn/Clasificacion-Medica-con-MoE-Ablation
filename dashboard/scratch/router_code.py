from monai.networks.blocks import PatchEmbed
import torch
import torch.nn as nn

class SwitchablePatchEmbed(nn.Module):
    """
    Switchable Patch Embedding (SPE) — Pasos D→I.
    Versión corregida y más robusta.
    """
    def __init__(self, embed_dim=192, patch_size_2d=16, patch_size_3d=8, in_channels_2d=3):
        super().__init__()
        self.embed_dim = embed_dim

        # D: Patch Embedding 2D
        self.patch_embed_2d = PatchEmbed(
            spatial_dims=2,
            in_chans=in_channels_2d,      # ← Correcto: in_chans
            patch_size=patch_size_2d,
            embed_dim=embed_dim           # ← Correcto: embed_dim
        )

        # E: Patch Embedding 3D
        self.patch_embed_3d = PatchEmbed(
            spatial_dims=3,
            in_chans=1,
            patch_size=patch_size_3d,
            embed_dim=embed_dim
        )

        # 1 (CLS) + max(14*14, 8**3) = 1 + 512 = 513 tokens
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, 513, embed_dim))

    def _patch_tokens_to_sequence(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """MONAI puede devolver [B,N,D] o tensores espaciales [B,H',W',D] / [B,D,H',W']."""
        x = patch_tokens
        if x.dim() == 3 and x.size(-1) == self.embed_dim:
            return x.squeeze(0) if x.size(0) == 1 else x.reshape(-1, self.embed_dim)
        if x.dim() >= 3 and x.size(-1) == self.embed_dim:
            return x.reshape(-1, self.embed_dim)
        if x.dim() == 4 and x.size(1) == self.embed_dim:
            x = x.flatten(2).transpose(1, 2).contiguous()
            return x.reshape(-1, self.embed_dim)
        if x.dim() == 5 and x.size(1) == self.embed_dim:
            x = x.flatten(2).transpose(1, 2).contiguous()
            return x.reshape(-1, self.embed_dim)
        raise RuntimeError(f'Forma de patch tokens no soportada: {tuple(patch_tokens.shape)}')

    def forward(self, batch_tensors):
        batch_size = len(batch_tensors)
        tokens_list = []

        for sample in batch_tensors:
            sample = sample.unsqueeze(0)  # [1, C, ...]

            if sample.ndim == 4:
                if sample.shape[1] == 1:
                    sample = sample.repeat(1, 3, 1, 1)
                patch_tokens = self.patch_embed_2d(sample)
            elif sample.ndim == 5:
                patch_tokens = self.patch_embed_3d(sample)
            else:
                raise ValueError(f'Tensor de entrada inválido (ndim={sample.ndim}): {tuple(sample.shape)}')

            seq = self._patch_tokens_to_sequence(patch_tokens)
            tokens_list.append(seq)

        # H: Padding
        padded = torch.nn.utils.rnn.pad_sequence(tokens_list, batch_first=True)

        # Máscara (True = válido)
        lengths = torch.tensor([t.size(0) for t in tokens_list], device=padded.device)
        max_len = padded.size(1)
        mask = torch.arange(max_len, device=padded.device)[None, :] < lengths[:, None]

        # I: CLS + Positional
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        final_tokens = torch.cat((cls_tokens, padded), dim=1)

        cls_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=mask.device)
        final_mask = torch.cat((cls_mask, mask), dim=1)

        final_tokens = final_tokens + self.pos_embed[:, :final_tokens.size(1), :]

        return final_tokens, final_mask

class VisionRouter(nn.Module):
    def __init__(self, embed_dim=192, num_experts=5, num_layers=12, pretrained=True):
        super().__init__()

        self.patch_embed = SwitchablePatchEmbed(embed_dim=embed_dim)

        # Usamos ViT-Tiny de timm (mucho más optimizado)
        self.vit = timm.create_model(
            'vit_tiny_patch16_224',
            pretrained=pretrained,
            num_classes=0,           # sin cabeza de clasificación
            global_pool='',          # no pooling
            img_size=224,            # solo referencia, no lo usamos realmente
        )

        # Reemplazamos el patch_embed original de timm por nuestro Switchable
        self.vit.patch_embed = nn.Identity()

        # Si queremos controlar el número de layers (opcional)
        if num_layers < 12:
            self.vit.blocks = self.vit.blocks[:num_layers]

        # Cabeza del router
        self.router_head = nn.Linear(embed_dim, num_experts)

    def forward(self, batch_tensors):
        # A → I: Patch + CLS + Positional
        x, mask = self.patch_embed(batch_tensors)   # [B, seq_len+1, 192]

        # timm ViT espera entrada sin CLS token + positional ya incluido
        # Como nosotros ya agregamos CLS y positional, pasamos directamente

        # Opción más limpia: forward manual solo de los bloques
        for blk in self.vit.blocks:
            x = blk(x)

        # K/L: Extraer CLS token (posición 0)
        cls_token = x[:, 0]                    # [B, 192]

        # M: Router head
        logits = self.router_head(cls_token)   # [B, 5]

        return logits, cls_token

print('VisionRouter definido (Pasos J→M).')
