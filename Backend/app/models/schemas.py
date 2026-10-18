from pydantic import BaseModel

class TextRequest(BaseModel):
    text: str


class IdentificationResponse(BaseModel):
    text: str 
    language: str 
    confidence: float 
    
class ErrorResponse(BaseModel):
    detail: str 
    
    
class TranslationResponse(BaseModel):
    text: str