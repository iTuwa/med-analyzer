from pydantic import BaseModel
from typing import List, Dict, Any

class RegionOfConcern(BaseModel):
    label: str
    bbox: List[int]  # [x_min, y_min, x_max, y_max]
    score: float

class AnalysisResponse(BaseModel):
    observation: str
    confidence: float
    confidence_display: str
    severity: str
    advice: List[str]
    generation_explanation: str
    regions_of_concern: List[RegionOfConcern]
    raw_model_response: Dict[str, Any]
