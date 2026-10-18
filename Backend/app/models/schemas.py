from pydantic import BaseModel, Field

class IdentificationResponse(BaseModel):
    text: str 
    language: str 
    confidence: float 
    
class ErrorResponse(BaseModel):
    detail: str 
    
    
class TranslationResponse(BaseModel):
    text: str