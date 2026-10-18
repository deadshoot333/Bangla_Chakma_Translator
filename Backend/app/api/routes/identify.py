import logging 

from fastapi import APIRouter, status, HTTPException

from app.models.schemas import ErrorResponse, IdentificationResponse
from app.services.inference import identification_service

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post(
    "/identify",
    response_model = IdentificationResponse,
    responses = {
        400: {"model": ErrorResponse, "description":"Invalid Text Format"},
        500: {"model": ErrorResponse, "description": "Model not loaded"}
    },
    summary = "Identify Bangla-Chakma text",
)

async def identify_language(text:str):
    try: 
        response = identification_service.identify(text) 
    except Exception as exc: 
        logger.exception("Inference failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Inference failed on the server. Check logs for details.",
        ) from exc 
        
        
    return response