"""
Diagnostic script to identify why the router always collapses to Expert 1 (ISIC).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Add the dashboard dir to path
dashboard_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, dashboard_dir)

import torch
import torch.nn.functional as F
import numpy as np
from huggingface_hub import hf_hub_download

HF_REPO = "Lucu1232004p/Proyecto-MoE-Pesos"

print("=" * 70)
print("DIAGNOSTIC: Router Collapse Investigation")
print("=" * 70)

# =====================================================================
# 1. Check .pth.zip vs .pth — are they the same format?
# =====================================================================
print("\n--- Step 1: Download and inspect both router files ---")

path_zip = hf_hub_download(HF_REPO, "router_professor_fase3_only.pth.zip")
path_pth = hf_hub_download(HF_REPO, "router_professor_fase3_only.pth")

print(f"  .pth.zip path: {path_zip}")
print(f"  .pth     path: {path_pth}")
print(f"  .pth.zip size: {os.path.getsize(path_zip):,} bytes")
print(f"  .pth     size: {os.path.getsize(path_pth):,} bytes")

# Check if .pth.zip is a regular ZIP (not PyTorch ZIP serialization)
import zipfile
is_regular_zip = zipfile.is_zipfile(path_zip)
print(f"  .pth.zip is a valid ZIP file: {is_regular_zip}")

if is_regular_zip:
    with zipfile.ZipFile(path_zip, 'r') as zf:
        names = zf.namelist()
        print(f"  ZIP contents: {names}")
        # Check if it contains a .pth file (meaning it's a manually zipped checkpoint)
        pth_inside = [n for n in names if n.endswith('.pth')]
        archive_inside = [n for n in names if 'archive/' in n or 'data.pkl' in n]
        print(f"  .pth files inside ZIP: {pth_inside}")
        print(f"  PyTorch archive entries: {archive_inside}")

# =====================================================================
# 2. Try loading both checkpoints
# =====================================================================
print("\n--- Step 2: Load both checkpoints ---")

try:
    ckpt_zip = torch.load(path_zip, map_location="cpu")
    print(f"  .pth.zip loaded OK. Type: {type(ckpt_zip)}")
    if isinstance(ckpt_zip, dict):
        print(f"  .pth.zip top-level keys: {list(ckpt_zip.keys())}")
        if 'model_state_dict' in ckpt_zip:
            sd_zip = ckpt_zip['model_state_dict']
            print(f"  .pth.zip state_dict keys: {len(sd_zip)}")
        else:
            sd_zip = ckpt_zip
            print(f"  .pth.zip IS the state_dict, keys: {len(sd_zip)}")
    else:
        print(f"  .pth.zip loaded as non-dict: {type(ckpt_zip)}")
        sd_zip = None
except Exception as e:
    print(f"  ERROR loading .pth.zip: {e}")
    sd_zip = None

try:
    ckpt_pth = torch.load(path_pth, map_location="cpu")
    print(f"  .pth loaded OK. Type: {type(ckpt_pth)}")
    if isinstance(ckpt_pth, dict):
        print(f"  .pth top-level keys: {list(ckpt_pth.keys())}")
        if 'model_state_dict' in ckpt_pth:
            sd_pth = ckpt_pth['model_state_dict']
            print(f"  .pth state_dict keys: {len(sd_pth)}")
        else:
            sd_pth = ckpt_pth
            print(f"  .pth IS the state_dict, keys: {len(sd_pth)}")
    else:
        print(f"  .pth loaded as non-dict: {type(ckpt_pth)}")
        sd_pth = None
except Exception as e:
    print(f"  ERROR loading .pth: {e}")
    sd_pth = None

# =====================================================================
# 3. Compare the two state dicts
# =====================================================================
if sd_zip is not None and sd_pth is not None:
    print("\n--- Step 3: Compare state dicts ---")
    keys_zip = set(sd_zip.keys())
    keys_pth = set(sd_pth.keys())
    print(f"  Keys only in .pth.zip: {keys_zip - keys_pth}")
    print(f"  Keys only in .pth: {keys_pth - keys_zip}")
    print(f"  Common keys: {len(keys_zip & keys_pth)}")
    
    # Check if the router_head weights are the same
    if 'router_head.weight' in sd_zip and 'router_head.weight' in sd_pth:
        w_zip = sd_zip['router_head.weight']
        w_pth = sd_pth['router_head.weight']
        diff = (w_zip - w_pth).abs().max().item()
        print(f"  router_head.weight max diff: {diff}")
        print(f"  router_head.weight shape (zip): {w_zip.shape}")
        print(f"  router_head.weight shape (pth): {w_pth.shape}")

# =====================================================================
# 4. Load the VisionRouter and check state_dict compatibility
# =====================================================================
print("\n--- Step 4: Load VisionRouter model ---")
from real_models import VisionRouter

router = VisionRouter(num_experts=5, pretrained=False)
model_keys = set(router.state_dict().keys())

# Try loading with strict=True
best_sd = sd_pth if sd_pth is not None else sd_zip
if best_sd is not None:
    ckpt_keys = set(best_sd.keys())
    missing = model_keys - ckpt_keys
    unexpected = ckpt_keys - model_keys
    print(f"  Model keys: {len(model_keys)}")
    print(f"  Checkpoint keys: {len(ckpt_keys)}")
    print(f"  Missing in checkpoint: {missing}")
    print(f"  Unexpected in checkpoint: {unexpected}")
    
    try:
        router.load_state_dict(best_sd, strict=True)
        print("  load_state_dict (strict=True): SUCCESS")
    except Exception as e:
        print(f"  load_state_dict (strict=True) FAILED: {e}")
        try:
            router.load_state_dict(best_sd, strict=False)
            print("  load_state_dict (strict=False): SUCCESS (with warnings above)")
        except Exception as e2:
            print(f"  load_state_dict (strict=False) FAILED: {e2}")

# =====================================================================
# 5. Test with a random image tensor - check raw logits
# =====================================================================
print("\n--- Step 5: Test router with synthetic tensors ---")
router.eval()

with torch.no_grad():
    # Test with a random 2D image (like ImageNet-normalized)
    fake_2d = torch.randn(3, 224, 224) * 0.2  # ~ImageNet-like range
    try:
        logits, cls_tok, attn = router(fake_2d)
        probs = F.softmax(logits, dim=-1).squeeze(0)
        print(f"  Random 2D image:")
        print(f"    Raw logits: {logits.squeeze(0).numpy()}")
        print(f"    Softmax probs: {probs.numpy()}")
        print(f"    Winner: Expert {probs.argmax().item()}")
        print(f"    CLS token stats: mean={cls_tok.mean().item():.4f}, std={cls_tok.std().item():.4f}")
    except Exception as e:
        print(f"  ERROR with 2D tensor: {e}")

    # Test with all-zeros
    zeros_2d = torch.zeros(3, 224, 224)
    try:
        logits, cls_tok, attn = router(zeros_2d)
        probs = F.softmax(logits, dim=-1).squeeze(0)
        print(f"  All-zeros 2D image:")
        print(f"    Raw logits: {logits.squeeze(0).numpy()}")
        print(f"    Softmax probs: {probs.numpy()}")
        print(f"    Winner: Expert {probs.argmax().item()}")
    except Exception as e:
        print(f"  ERROR with zeros tensor: {e}")

    # Test with all-ones
    ones_2d = torch.ones(3, 224, 224)
    try:
        logits, cls_tok, attn = router(ones_2d)
        probs = F.softmax(logits, dim=-1).squeeze(0)
        print(f"  All-ones 2D image:")
        print(f"    Raw logits: {logits.squeeze(0).numpy()}")
        print(f"    Softmax probs: {probs.numpy()}")
        print(f"    Winner: Expert {probs.argmax().item()}")
    except Exception as e:
        print(f"  ERROR with ones tensor: {e}")

# =====================================================================
# 6. Inspect router_head weights directly
# =====================================================================
print("\n--- Step 6: Inspect router_head weights ---")
if best_sd is not None:
    rh_w = best_sd.get('router_head.weight')
    rh_b = best_sd.get('router_head.bias')
    if rh_w is not None:
        print(f"  router_head.weight shape: {rh_w.shape}")
        print(f"  router_head.weight norms per expert:")
        for i in range(rh_w.shape[0]):
            print(f"    Expert {i}: norm={rh_w[i].norm().item():.4f}, mean={rh_w[i].mean().item():.6f}")
    if rh_b is not None:
        print(f"  router_head.bias: {rh_b.numpy()}")

# =====================================================================
# 7. Check if .pth.zip needs extraction (like the expert ZIP files)
# =====================================================================
print("\n--- Step 7: Check if .pth.zip is a ZIP-wrapped .pth ---")
if is_regular_zip:
    with zipfile.ZipFile(path_zip, 'r') as zf:
        names = zf.namelist()
        pth_inside = [n for n in names if n.endswith('.pth')]
        if pth_inside:
            print(f"  YES! .pth.zip contains: {pth_inside}")
            print(f"  The router loading code does NOT extract it!")
            print(f"  This is likely the bug — torch.load is loading the ZIP metadata,")
            print(f"  not the actual .pth checkpoint inside.")
            
            # Try extracting and loading
            import tempfile
            with tempfile.TemporaryDirectory() as tmpdir:
                zf.extractall(tmpdir)
                inner_pth = os.path.join(tmpdir, pth_inside[0])
                try:
                    inner_ckpt = torch.load(inner_pth, map_location="cpu")
                    print(f"\n  Extracted .pth loaded OK. Type: {type(inner_ckpt)}")
                    if isinstance(inner_ckpt, dict):
                        print(f"  Top-level keys: {list(inner_ckpt.keys())}")
                        inner_sd = inner_ckpt.get('model_state_dict', inner_ckpt)
                        print(f"  State dict keys: {len(inner_sd)}")
                        
                        # Compare with direct load
                        if sd_zip is not None:
                            for key in ['router_head.weight', 'router_head.bias']:
                                if key in inner_sd and key in sd_zip:
                                    diff = (inner_sd[key] - sd_zip[key]).abs().max().item()
                                    print(f"  {key} diff (inner vs direct): {diff}")
                except Exception as e:
                    print(f"  ERROR loading extracted .pth: {e}")
        else:
            print(f"  No .pth files inside ZIP — it's a PyTorch ZIP serialization, not a wrapped file.")

print("\n" + "=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)
