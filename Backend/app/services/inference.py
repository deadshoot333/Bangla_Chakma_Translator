import logging 
import joblib 
import time 
import torch
from app.models.schemas import IdentificationResponse, TranslationResponse
from app.core.state import models

logger = logging.getLogger(__name__)

class IdentificationService:
        
    def identify(self,text:str)->IdentificationResponse:
        model = models["naive_bayes"] 
        vectorizer = models["vectorizer"]
        if model is None: 
            raise RuntimeError("Model is not loaded")
        
        start = time.perf_counter()
        features = vectorizer.transform([text])
        language = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        confidence = float(max(probabilities))
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        return IdentificationResponse(
            text = text,
            language = language, 
            confidence=confidence
        )

class TranslationService_Bangla2Chakma:
    
    def Bangla2Chakma(self,text:str)->TranslationResponse:
        model = models["model_ban2chak"]
        tokenizer = models["tokenizer_ban2chak"]
        device = models["device"]
        print(f"Model Initialized on: {device}")
        
        input_text = text 
        inputs = tokenizer(input_text, return_tensors="pt",truncation=True).to(device)
        outputs = model.generate(
        **inputs,
        max_length=128,
        num_beams=4
        )
        return TranslationResponse(
            text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        )
        
class TranslationService_Chakma2Bangla:
    
    def Chakma2Bangla(self,text:str)->TranslationResponse:
        model = models["model_chak2ban"]
        tokenizer = models["tokenizer_chak2ban"]
        device = models["device"]
        print(f"Model Initialized on: {device}")
        input_text = text 
        inputs = tokenizer(input_text, return_tensors="pt",truncation=True).to(device)
        outputs = model.generate(
        **inputs,
        max_length=128,
        num_beams=4
        )
        return TranslationResponse(
            text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        )        


        
identification_service = IdentificationService()
translation_service_ban2chak = TranslationService_Bangla2Chakma()
translation_service_chak2ban = TranslationService_Chakma2Bangla()