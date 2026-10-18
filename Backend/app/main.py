import logging 
from fastapi import FastAPI, status 
from fastapi.middleware.cors import CORSMiddleware 

from app.api.routes.identify import router as identify_router
from app.api.routes.translate import router as translate_router

logger = logging.getLogger(__name__)


app = FastAPI(
    title = "demo",
    version = "0.0.1",
    description="Chakma Translator and Detector"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/",status_code=status.HTTP_200_OK)
def home():
    return "Welcome to Chakma Translator"

app.include_router(identify_router, tags=["Inference"])
app.include_router(translate_router,tags=["Inference"])
