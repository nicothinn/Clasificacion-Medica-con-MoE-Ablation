"""
Diagnostic #3: Quick test - try router WITHOUT ImageNet norm on a real image.
Also test with the .pth version (not .pth.zip).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

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
print("DIAGNOSTIC #3: Real image test")
print("=" * 70)

# Load .pth.zip router
path_zip = hf_hub_download(HF_REPO, "router_professor_fase3_only.pth.zip")
ckpt_zip = torch.load(path_zip, map_location="cpu")
router_zip = VisionRouter(num_experts=5, pretrained=False)
router_zip.load_state_dict(ckpt_zip["model_state_dict"])
router_zip.eval()

# Load .pth router
path_pth = hf_hub_download(HF_REPO, "router_professor_fase3_only.pth")
ckpt_pth = torch.load(path_pth, map_location="cpu")
router_pth = VisionRouter(num_experts=5, pretrained=False)
router_pth.load_state_dict(ckpt_pth["model_state_dict"])
router_pth.eval()

# Preprocessor
prep = AdaptivePreprocessor()
transform = T.Compose([T.Resize((224, 224)), T.ToTensor()])

# Find real test images in the project
test_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
possible_dirs = [
    os.path.expanduser("~\\Desktop"),
    os.path.expanduser("~\\Downloads"),
    test_dir,
]

# Also try to find any image files in the workspace
print("\nSearching for test images...")
test_images = {}
for pd in possible_dirs:
    if os.path.exists(pd):
        for f in os.listdir(pd):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')) and any(
                kw in f.lower() for kw in ['osteo', 'knee', 'nih', 'xray', 'isic', 'chest', 'skin']
            ):
                test_images[f] = os.path.join(pd, f)

if not test_images:
    print("  No real test images found. Using extensive synthetic tests.")
    
    # Create very realistic synthetic test images
    np.random.seed(42)
    
    # Chest X-ray style (dark lung fields, bright ribs, grayscale)
    chest = np.zeros((224, 224, 3), dtype=np.float32)
    chest[:, :] = 0.08  # mostly dark
    # Mediastinum (center bright area)
    chest[60:180, 90:130] = 0.5
    # Ribs (horizontal bright lines)
    for y in range(40, 200, 20):
        chest[y:y+3, 30:200] = 0.6
    # Add noise
    chest += np.random.normal(0, 0.03, chest.shape).astype(np.float32)
    chest = np.clip(chest, 0, 1)
    
    # Knee X-ray (bright bone, medium soft tissue)
    knee = np.zeros((224, 224, 3), dtype=np.float32)
    knee[:, :] = 0.15  # dark background
    # Femur (upper bone)
    knee[30:120, 60:170] = 0.55
    # Tibia (lower bone)
    knee[130:210, 60:170] = 0.55
    # Joint space (dark gap)
    knee[118:132, 70:160] = 0.1
    # Add noise
    knee += np.random.normal(0, 0.03, knee.shape).astype(np.float32)
    knee = np.clip(knee, 0, 1)
    
    # Skin lesion (colorful, has color variation)
    skin = np.zeros((224, 224, 3), dtype=np.float32)
    skin[:, :, 0] = 0.72  # skin R
    skin[:, :, 1] = 0.55  # skin G
    skin[:, :, 2] = 0.45  # skin B
    # Dark melanoma-like region
    cx, cy = 112, 112
    for x in range(224):
        for y in range(224):
            dist = np.sqrt((x-cx)**2 + (y-cy)**2)
            if dist < 40:
                skin[x, y] = [0.2, 0.1, 0.08]
            elif dist < 50:
                skin[x, y] = [0.4, 0.25, 0.2]
    skin += np.random.normal(0, 0.02, skin.shape).astype(np.float32)
    skin = np.clip(skin, 0, 1)
    
    synth_images = {
        "Synthetic Chest X-ray": torch.from_numpy(chest).permute(2, 0, 1),
        "Synthetic Knee X-ray": torch.from_numpy(knee).permute(2, 0, 1),
        "Synthetic Skin Lesion": torch.from_numpy(skin).permute(2, 0, 1),
    }
    
    print("\n" + "=" * 70)
    print("RESULTS: Comparing normalizations and checkpoints")
    print("=" * 70)
    
    with torch.no_grad():
        for img_name, tensor_raw in synth_images.items():
            tensor_imagenet = AdaptivePreprocessor.apply_imagenet_norm(tensor_raw)
            
            print(f"\n--- {img_name} ---")
            print(f"  Input range: [{tensor_raw.min():.3f}, {tensor_raw.max():.3f}]")
            print(f"  After ImageNet norm: [{tensor_imagenet.min():.3f}, {tensor_imagenet.max():.3f}]")
            
            for ckpt_name, r in [(".pth.zip (val_acc=88.5%)", router_zip), 
                                  (".pth (val_acc=77.6%)", router_pth)]:
                for norm_name, inp in [("NO norm (raw [0,1])", tensor_raw),
                                       ("ImageNet norm", tensor_imagenet)]:
                    logits, cls, _ = r(inp)
                    probs = F.softmax(logits, dim=-1).squeeze(0).numpy()
                    winner_idx = probs.argmax()
                    print(f"  {ckpt_name:30s} + {norm_name:20s} -> "
                          f"Winner: {experts[winner_idx]:15s} ({probs[winner_idx]*100:5.1f}%) | "
                          f"logits: [{', '.join(f'{x:6.1f}' for x in logits.squeeze(0).numpy())}]")

else:
    print(f"  Found {len(test_images)} images: {list(test_images.keys())}")
    
    with torch.no_grad():
        for fname, fpath in test_images.items():
            img = Image.open(fpath).convert("RGB")
            tensor_raw = transform(img)
            tensor_imagenet = AdaptivePreprocessor.apply_imagenet_norm(tensor_raw)
            
            print(f"\n--- {fname} ---")
            
            for ckpt_name, r in [(".pth.zip", router_zip), (".pth", router_pth)]:
                for norm_name, inp in [("NO norm", tensor_raw), ("ImageNet", tensor_imagenet)]:
                    logits, _, _ = r(inp)
                    probs = F.softmax(logits, dim=-1).squeeze(0).numpy()
                    winner_idx = probs.argmax()
                    print(f"  {ckpt_name} + {norm_name}: "
                          f"Winner={experts[winner_idx]} ({probs[winner_idx]*100:.1f}%)")

print("\n" + "=" * 70)
print("DIAGNOSTIC #3 COMPLETE")  
print("=" * 70)
