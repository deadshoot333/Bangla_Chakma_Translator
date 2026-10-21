import logging 

from fastapi import APIRouter, status, HTTPException
from app.models.schemas import ErrorResponse, TextRequest, TranslationResponse
from app.services.inference import translation_service_ban2chak, translation_service_chak2ban

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post(
    "/translate/ban2chak",
    response_model = TranslationResponse,
    responses = {
        400: {"model": ErrorResponse, "description":"Invalid Text Format"},
        500: {"model": ErrorResponse, "description": "Model not loaded"}
    },
    summary = "Translate Bangla to Chakma",
)

async def translate_language(request: TextRequest):
    try: 
        response = translation_service_ban2chak.Bangla2Chakma(request.text) 
    except Exception as exc: 
        logger.exception("Inference failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Inference failed on the server. Check logs for details.",
        ) from exc 
        
        
    return response

@router.post(
    "/translate/chak2ban",
    response_model = TranslationResponse,
    responses = {
        400: {"model": ErrorResponse, "description":"Invalid Text Format"},
        500: {"model": ErrorResponse, "description": "Model not loaded"}
    },
    summary = "Translate Chakma to Bangla",
)

async def translate_language(request: TextRequest):
    try: 
        response = translation_service_chak2ban.Chakma2Bangla(request.text) 
    except Exception as exc: 
        logger.exception("Inference failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Inference failed on the server. Check logs for details.",
        ) from exc 
        
        
    return response