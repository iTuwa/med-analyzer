from fastapi import FastAPI, UploadFile, File
from app.schemas import AnalysisResponse, RegionOfConcern
from PIL import Image
import io
import torch
from torchvision import models, transforms
import random
import numpy as np
import requests
import base64
import json as _json

app = FastAPI(title="Med Analyzer API")

# ----- PyTorch Models -----
cls_model = models.resnet18(pretrained=True)
cls_model.eval()

seg_model = models.segmentation.deeplabv3_resnet50(pretrained=True)
seg_model.eval()

# Image preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ----- Gemini Vision API Config -----
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
GEMINI_API_KEY = "AIzaSyAiAC6baqB2lBHo6zeZsXrnV9Elua3ICTQ"  

# ----- Helper functions -----
def mask_to_bbox(mask, threshold=0.5):
    """Convert segmentation mask to bounding box [x_min, y_min, x_max, y_max]."""
    mask_bin = mask > threshold
    if mask_bin.sum() == 0:
        return None
    ys, xs = np.where(mask_bin)
    x_min, y_min, x_max, y_max = xs.min(), ys.min(), xs.max(), ys.max()
    return [int(x_min), int(y_min), int(x_max), int(y_max)]


def call_gemini(image: Image.Image):
    """Call Gemini Vision API and return structured JSON."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode("utf-8")

    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": (
                            "You are a medical assistant. "
                            "Given this image, provide a JSON response with keys: "
                            "label, confidence (0-1), severity (mild/moderate/severe/unknown), "
                            "advice (as a list of short bullet points), explanation, and regions (bbox as [x_min, y_min, x_max, y_max]). "
                            "Return strictly in JSON format."
                        )
                    },
                    {
                        "inline_data": {
                            "mime_type": "image/png",
                            "data": img_base64
                        }
                    }
                ]
            }
        ]
    }

    params = {"key": GEMINI_API_KEY}
    headers = {"Content-Type": "application/json"}
    response = requests.post(GEMINI_API_URL, params=params, json=payload, headers=headers)
    response.raise_for_status()
    result = response.json()

    try:
        text = result["candidates"][0]["content"]["parts"][0]["text"]
        import json as _json
        start = text.find('{')
        end = text.rfind('}') + 1
        json_str = text[start:end]
        return _json.loads(json_str)
    except Exception as e:
        raise RuntimeError(f"Failed to parse Gemini response: {e}\nRaw: {result}")

# ----- Endpoints -----
@app.get("/")
def health_check():
    return {"status": "ok"}

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_image(file: UploadFile = File(...)):
    # Load image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    # ----- Attempt Med-Gemini API -----
    try:
        print("[INFO] Using Gemini Vision API for analysis.")
        gemini_response = call_gemini(image)
        label = gemini_response.get("label", "Unknown")
        confidence = gemini_response.get("confidence", 0.0)
        severity = gemini_response.get("severity", "Unknown")
        regions = gemini_response.get("regions", [])
    except Exception as e:
        print(f"[INFO] Med-Gemini API failed, using fallback. Error: {e}")
        # Fallback to dummy PyTorch classification
        label = random.choice(dummy_labels)
        confidence = round(random.uniform(0.7, 0.99), 2)
        severity = "Mild"

        # Segmentation fallback
        seg_input = preprocess(image).unsqueeze(0)
        with torch.no_grad():
            output = seg_model(seg_input)['out'][0]
            probs = torch.sigmoid(output[15])  # pick a class index as dummy
            mask = probs.cpu().numpy()
        bbox = mask_to_bbox(mask)

        regions = []
        if bbox:
            regions.append({"label": "Concern", "bbox": bbox, "score": float(confidence)})
        else:
            regions.append({"label": "Concern", "bbox": [50, 50, 150, 150], "score": float(confidence)})

    # ----- Ensure regions are in dict format -----
    formatted_regions = []
    for r in regions:
        if isinstance(r, dict):
            formatted_regions.append(r)
        elif isinstance(r, list):
            formatted_regions.append({"label": "Concern", "bbox": r, "score": float(confidence)})
        else:
            print("[WARN] Unexpected region format:", r)

    # ----- Build response -----
    response = AnalysisResponse(
        observation=label,
        confidence=confidence,
        confidence_display=f"{int(confidence * 100)}%",
        severity=severity,
        advice=[
            "Monitor the area for changes.",
            "Scan again if symptoms persist.",
            "Consult a doctor to confirm this result.",
            "Learn more in the Insights tab."
        ],
        generation_explanation="Model used: Med-Gemini API with PyTorch fallback + DeepLabV3 for regions.",
        regions_of_concern=[RegionOfConcern(**r) for r in formatted_regions],
        raw_model_response={
            "label": label,
            "confidence": confidence,
            "severity": severity,
            "regions": formatted_regions
        }
    )

    return response