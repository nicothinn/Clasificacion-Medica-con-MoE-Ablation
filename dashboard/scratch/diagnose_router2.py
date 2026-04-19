"""
Diagnostic #2 (fixed): Test router with real-like images + version checks.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
os.environ["PYTHONIOENCODING"] = "utf-8"

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as T
from huggingface_hub import hf_hub_download

from real_models import VisionRouter
from preprocessing import AdaptivePreprocessor

HF_REPO = "Lucu1232004p/Proyecto-MoE-Pesos"
experts = ["NIH ChestXray", "ISIC 2019", "Knee Osteo", "LUNA16", "Pancreas"]

print("=" * 70)
print("DIAGNOSTIC #2: Normalization + Version Check")
print("=" * 70)

# =====================================================================
# 0. Version checks
# =====================================================================
import timm
print(f"\n  timm version: {timm.__version__}")
try:
    import monai
    print(f"  monai version: {monai.__version__}")
except:
    print("  monai: NOT INSTALLED")
print(f"  torch version: {torch.__version__}")
import torchvision
print(f"  torchvision version: {torchvision.__version__}")

# =====================================================================
# 1. Load router
# =====================================================================
path_zip = hf_hub_download(HF_REPO, "router_professor_fase3_only.pth.zip")
ckpt = torch.load(path_zip, map_location="cpu")

router = VisionRouter(num_experts=5, pretrained=False)
router.load_state_dict(ckpt["model_state_dict"])
router.eval()

# Also load the .pth version for comparison
path_pth = hf_hub_download(HF_REPO, "router_professor_fase3_only.pth")
ckpt_pth = torch.load(path_pth, map_location="cpu")
router_v2 = VisionRouter(num_experts=5, pretrained=False)
router_v2.load_state_dict(ckpt_pth["model_state_dict"])
router_v2.eval()

print("\nCheckpoint metadata (.pth.zip):")
print(f"  mode: {ckpt.get('mode')}")
print(f"  val_acc: {ckpt.get('best_meta', {}).get('val_acc')}")

print("\nCheckpoint metadata (.pth):")
print(f"  mode: {ckpt_pth.get('mode')}")
print(f"  val_acc: {ckpt_pth.get('best_meta', {}).get('val_acc')}")

# =====================================================================
# 2. Test with different normalization strategies
# =====================================================================
print("\n" + "=" * 70)
print("TEST: Synthetic images - different normalization strategies")
print("=" * 70)

with torch.no_grad():
    # Create a realistic X-ray image (grayscale, mostly dark)
    gray_val = 0.25
    xray = torch.full((3, 224, 224), gray_val)
    # Add bright bone regions
    xray[:, 40:180, 70:160] = 0.65
    # Add darker lung regions
    xray[:, 60:160, 85:145] = 0.15
    
    # Create a realistic skin image (colorful)
    skin = torch.zeros(3, 224, 224)
    skin[0] = 0.65  # reddish skin
    skin[1] = 0.45
    skin[2] = 0.35
    skin[:, 70:150, 70:150] = torch.tensor([0.3, 0.2, 0.15]).reshape(3,1,1)
    
    test_images = {
        "Synthetic X-ray (gray)": xray,
        "Synthetic skin lesion": skin,
        "All black (0.0)": torch.zeros(3, 224, 224),
        "Mid gray (0.5)": torch.full((3, 224, 224), 0.5),
        "All white (1.0)": torch.ones(3, 224, 224),
    }
    
    for name, img in test_images.items():
        # Without ImageNet norm
        logits_raw, _, _ = router(img)
        probs_raw = F.softmax(logits_raw, dim=-1).squeeze(0).numpy()
        
        # With ImageNet norm
        img_norm = AdaptivePreprocessor.apply_imagenet_norm(img)
        logits_norm, _, _ = router(img_norm)
        probs_norm = F.softmax(logits_norm, dim=-1).squeeze(0).numpy()
        
        print(f"\n  {name}:")
        print(f"    NO norm  -> Winner: {experts[probs_raw.argmax()]:15s} | "
              f"logits: [{', '.join(f'{x:7.2f}' for x in logits_raw.squeeze(0).numpy())}]")
        print(f"    ImageNet -> Winner: {experts[probs_norm.argmax()]:15s} | "
              f"logits: [{', '.join(f'{x:7.2f}' for x in logits_norm.squeeze(0).numpy())}]")

# =====================================================================
# 3. Check if the issue is norm: try a real-like distribution
# =====================================================================
print("\n" + "=" * 70)
print("TEST: Real image pixel distributions")
print("=" * 70)

with torch.no_grad():
    # Simulate what a real X-ray looks like after T.ToTensor()
    # X-rays are mostly dark with some bright regions
    np.random.seed(42)
    xray_real = np.random.normal(0.3, 0.15, (224, 224)).clip(0, 1).astype(np.float32)
    # All 3 channels same (grayscale)
    xray_tensor = torch.from_numpy(xray_real).unsqueeze(0).repeat(3, 1, 1)
    
    # Test both versions (.pth.zip and .pth) with and without norm
    for router_name, r in [(".pth.zip", router), (".pth", router_v2)]:
        print(f"\n  Router: {router_name}")
        
        logits_raw, cls_raw, _ = r(xray_tensor)
        probs_raw = F.softmax(logits_raw, dim=-1).squeeze(0).numpy()
        
        xray_norm = AdaptivePreprocessor.apply_imagenet_norm(xray_tensor)
        logits_norm, cls_norm, _ = r(xray_norm)
        probs_norm = F.softmax(logits_norm, dim=-1).squeeze(0).numpy()
        
        print(f"    Simulated X-ray NO norm:")
        print(f"      Winner: {experts[probs_raw.argmax()]} ({probs_raw.max()*100:.1f}%)")
        print(f"      Probs: [{', '.join(f'{p:.3f}' for p in probs_raw)}]")
        print(f"      CLS stats: mean={cls_raw.mean().item():.3f} std={cls_raw.std().item():.3f}")
        
        print(f"    Simulated X-ray WITH ImageNet norm:")
        print(f"      Winner: {experts[probs_norm.argmax()]} ({probs_norm.max()*100:.1f}%)")
        print(f"      Probs: [{', '.join(f'{p:.3f}' for p in probs_norm)}]")
        print(f"      CLS stats: mean={cls_norm.mean().item():.3f} std={cls_norm.std().item():.3f}")

# =====================================================================
# 4. Check the SwitchablePatchEmbed output directly
# =====================================================================
print("\n" + "=" * 70)
print("TEST: SwitchablePatchEmbed output analysis")
print("=" * 70)

with torch.no_grad():
    xray_norm = AdaptivePreprocessor.apply_imagenet_norm(xray_tensor)
    
    # Get patch embed output
    x = router.patch_embed(xray_norm)
    print(f"  Patch embed output shape: {x.shape}")
    print(f"  Patch embed output stats: mean={x.mean().item():.4f}, std={x.std().item():.4f}")
    print(f"  Patch embed CLS token (pos 0): mean={x[0,0].mean().item():.4f}")
    
    # Run through ViT blocks one by one
    for i, blk in enumerate(router.vit.blocks):
        x = blk(x)
        if i == 0 or i == len(router.vit.blocks)-1:
            print(f"  After block {i}: mean={x.mean().item():.4f}, std={x.std().item():.4f}, "
                  f"CLS mean={x[0,0].mean().item():.4f}")
    
    # Before and after norm
    x_prenorm = x.clone()
    x_normed = router.vit.norm(x)
    cls_prenorm = x_prenorm[:, 0]
    cls_normed = x_normed[:, 0]
    
    print(f"\n  CLS before vit.norm: mean={cls_prenorm.mean().item():.4f}, std={cls_prenorm.std().item():.4f}")
    print(f"  CLS after vit.norm:  mean={cls_normed.mean().item():.4f}, std={cls_normed.std().item():.4f}")
    
    logits_prenorm = router.router_head(cls_prenorm)
    logits_normed = router.router_head(cls_normed)
    
    probs_prenorm = F.softmax(logits_prenorm, dim=-1).squeeze(0).numpy()
    probs_normed = F.softmax(logits_normed, dim=-1).squeeze(0).numpy()
    
    print(f"\n  router_head(CLS_prenorm): Winner={experts[probs_prenorm.argmax()]} ({probs_prenorm.max()*100:.1f}%)")
    print(f"    logits: [{', '.join(f'{x:.2f}' for x in logits_prenorm.squeeze(0).numpy())}]")
    print(f"  router_head(CLS_normed):  Winner={experts[probs_normed.argmax()]} ({probs_normed.max()*100:.1f}%)")
    print(f"    logits: [{', '.join(f'{x:.2f}' for x in logits_normed.squeeze(0).numpy())}]")

print("\n" + "=" * 70)
print("DIAGNOSTIC #2 COMPLETE")
print("=" * 70)
