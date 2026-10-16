import logging 
import joblib 
import time 
from app.models.schemas import IdentificationResponse

logger = logging.getLogger(__name__)

class IdentificationService:
    def __init__(self): 
        model = joblib.load("weights/Bangla_Chakma_Identifier_Naive_Bayes.pkl") 
        vectorizer = joblib.load("weights/count_vectorizer.pkl")
        self._model = model 
        self._vectorizer = vectorizer 
        
    def identify(self,text:str)->IdentificationResponse:
        if self._model is None: 
            raise RuntimeError("Model is not loaded")
        
        start = time.perf_counter()
        features = self._vectorizer.transform([text])
        language = self._model.predict(features)[0]
        probabilities = self._model.predict_proba(features)[0]
        confidence = float(max(probabilities))
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        return IdentificationResponse(
            text = text,
            language = language, 
            confidence=confidence
        )
        
identification_service = IdentificationService()