import requests

print("Sending request to /api/predict...")
try:
    resp = requests.post("http://127.0.0.1:8000/api/predict", files={"file": ("test.png", b"fake image bytes")}, timeout=60)
    print("Response code:", resp.status_code)
    print("Response text:", resp.text)
except Exception as e:
    print("Error:", e)
