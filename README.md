# Med-Analyzer

A small Python service that accepts uploaded dermatology or eye images, queries a medical multimodal model (Med-Gemini) for diagnosis, and returns structured JSON with confidence, severity, advice, and regions of concern.

Built with **FastAPI** and **PyTorch** for local model fallback.

---

## **Table of Contents**

1. [Requirements](#requirements)
2. [Setup](#setup)
3. [Running the Server](#running-the-server)
4. [Testing the API](#testing-the-api)
5. [Example curl Requests](#example-curl-requests)
6. [Sample Images](#sample-images)
7. [JSON Schema](#json-schema)
8. [Model Limitations & Verification](#model-limitations--verification)

---

## **Requirements**

* Python 3.10+
* Packages:

```
fastapi
uvicorn
torch
torchvision
Pillow
requests
pydantic
numpy
pytest
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## **Setup**

1. Clone the repository:

```bash
git clone <your_repo_url>
cd med-analyzer
```

2. Set your **Med-Gemini API key** as an environment variable:

```bash
export MED_GEMINI_API_KEY="your_api_key_here"
```

> On Windows (PowerShell):
>
> ```powershell
> setx MED_GEMINI_API_KEY "your_api_key_here"
> ```

3. Ensure you have a `samples/` folder with at least one test image:

```
samples/sample1.jpg
```

---

## **Running the Server**

Start the FastAPI server:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

* Swagger UI: `http://localhost:8000/docs`
* Health check: `http://localhost:8000/` returns:

```json
{"status": "ok"}
```

---

## **Testing the API**

### **Unit Tests**

1. Make sure you have `pytest` installed:

```bash
pip install pytest
```

2. Run the tests:

```bash
export PYTHONPATH=$(pwd)
pytest tests/
```

* You should see `..` indicating tests passed.
* Tests include checking `/analyze` endpoint with a sample image.

---

## **Example curl Requests**

```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@samples/sample1.jpg"
```

* Replace `sample1.jpg` with any JPG/PNG image in `samples/`.

---

## **Sample Images**

* Store sample dermatology or eye images in the `samples/` folder for testing.
* These are used in **unit tests** and **curl requests**.

---

## **JSON Schema**

Example JSON response:

```json
{
  "observation": "Possible Contact Dermatitis",
  "confidence": 0.87,
  "confidence_display": "87%",
  "severity": "Mild",
  "advice": [
    "Monitor the area for changes.",
    "Scan again if symptoms persist.",
    "Consult a doctor to confirm this result.",
    "Learn more in the Insights tab."
  ],
  "generation_explanation": "Model used: Med-Gemini API with PyTorch fallback + DeepLabV3 for regions.",
  "regions_of_concern": [
    {
      "label": "Concern",
      "bbox": [50, 50, 150, 150],
      "score": 0.87
    }
  ],
  "raw_model_response": {
    "label": "Possible Contact Dermatitis",
    "confidence": 0.87,
    "severity": "Mild",
    "regions": [
      {
        "label": "Concern",
        "bbox": [50, 50, 150, 150],
        "score": 0.87
      }
    ]
  }
}
```

---

## **Model Limitations & Verification**

### **Model Limitations**

1. Predictions are **probabilistic**, not guaranteed correct.
2. **Not a replacement for a licensed medical professional**.
3. Local PyTorch segmentation may not be fine-tuned for specific conditions.
4. Image quality affects prediction accuracy.

### **Verification Steps**

1. Keep `raw_model_response` for audit.
2. Take multiple images under good lighting.
3. Confirm results with a **medical professional**.
4. Run **pytest** regularly to verify endpoint functionality.
5. System falls back to local PyTorch predictions if Med-Gemini is unavailable.
