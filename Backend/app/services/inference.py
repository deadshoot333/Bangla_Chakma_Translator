import logging 
import joblib 
import time 
import torch
from app.models.schemas import IdentificationResponse, TranslationResponse
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


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

class TranslationService_Bangla2Chakma:
    def __init__(self):
        model_path = "weights/Fine_tuned_BanglaT5"
        self._model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        self._tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Model Initialized on: {self._device}")
        self._model.to(self._device)

    
    def Bangla2Chakma(self,text:str)->TranslationResponse:
        input_text = text 
        inputs = self._tokenizer(input_text, return_tensors="pt",truncation=True).to(self._device)
        outputs = self._model.generate(
        **inputs,
        max_length=128,
        num_beams=4
        )
        return TranslationResponse(
            text = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
            )
        
        


        
identification_service = IdentificationService()
translation_service_ban2chak = TranslationService_Bangla2Chakma()