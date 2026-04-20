import os
import io
import base64
from io import BytesIO
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
import numpy as np
from PIL import Image

from moe_inference import MoEInferenceEngine

app = FastAPI(title="MoE Medical Vision Router API")

# Ensure static directory exists
os.makedirs("static", exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_root():
    return FileResponse("static/index.html")

# Initialize engine globally
engine = None

@app.on_event("startup")
async def startup_event():
    global engine
    print("Initializing MoE Inference Engine...")
    engine = MoEInferenceEngine(use_mock=False)
    print("MoE Inference Engine initialized successfully.")


def pil_to_base64(img):
    if img is None:
        return ""
    if not isinstance(img, Image.Image):
        img = Image.fromarray(np.array(img))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

from typing import List

@app.post("/api/predict")
def predict(files: List[UploadFile] = File(...), source: str = Form("unknown")):
    global engine
    if engine is None:
        return JSONResponse(content={"success": False, "error": "Engine not initialized"}, status_code=500)

    try:
        result = engine.run(files, source=source)
        
        response_data = {
            "success": True,
            "original_shape": list(result.original_shape),
            "processed_shape": list(result.processed_shape),
            "is_3d": result.is_3d,
            "preprocess_ms": result.preprocess_ms,
            "router_ms": result.router_ms,
            "expert_ms": result.expert_ms,
            "total_ms": result.total_ms,
            "gating_scores": result.gating_scores.tolist(),
            "expert_id": result.expert_id,
            "expert_name": result.expert_name,
            "expert_arch": result.expert_arch,
            "expert_dataset": result.expert_dataset,
            "class_label": result.class_label,
            "confidence": result.confidence,
            "all_class_probs": result.all_class_probs.tolist() if isinstance(result.all_class_probs, np.ndarray) else list(result.all_class_probs),
            "class_names": result.class_names,
            "is_ood": bool(result.is_ood),
            "entropy": float(result.entropy),
            "ood_threshold": float(result.ood_threshold),
            "display_image": pil_to_base64(result.display_image),
            "heatmap_image": pil_to_base64(result.heatmap_image) if result.heatmap_image is not None else ""
        }
        return JSONResponse(content=response_data)
    except Exception as e:
        return JSONResponse(content={"success": False, "error": str(e)}, status_code=500)



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
