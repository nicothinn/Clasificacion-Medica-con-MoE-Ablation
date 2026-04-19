import sys
import logging
logging.basicConfig(level=logging.DEBUG)

print("Importing MoEInferenceEngine...")
from moe_inference import MoEInferenceEngine
import io
import time

print("Initializing Engine...")
engine = MoEInferenceEngine(use_mock=False)
print("Engine Initialized.")

print("Creating fake file object...")
file_obj = io.BytesIO(b'fake_file_content')
file_obj.name = "fake.png"

# Wait, fake content will fail in PIL.Image.open
# Let's create a real fake PNG image
from PIL import Image
import numpy as np
img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
buf = io.BytesIO()
img.save(buf, format="PNG")
buf.seek(0)
buf.name = "fake.png"

print("Running engine...")
try:
    start = time.time()
    result = engine.run(buf, source="unknown")
    print(f"Finished in {time.time() - start:.2f}s")
except Exception as e:
    print("Exception during engine.run:", e)
