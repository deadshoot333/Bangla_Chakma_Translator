import logging 
from fastapi import FastAPI, status 
from fastapi.middleware.cors import CORSMiddleware 
from contextlib import asynccontextmanager
import joblib
import torch 
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from app.api.routes.identify import router as identify_router
from app.api.routes.translate import router as translate_router
from app.core.state import models

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # This code runs BEFORE the server starts accepting requests
    print("Loading machine learning model...")
    models["naive_bayes"] = joblib.load("weights/Bangla_Chakma_Identifier_Naive_Bayes.pkl")
    models["vectorizer"] = joblib.load("weights/count_vectorizer.pkl")
    print(f"Loading deep learning Models...")
    model_path = "weights/Fine_tuned_BanglaT5"
    models["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    models["model_ban2chak"] = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(models["device"])
    models["tokenizer_ban2chak"] = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model_path_1 = "weights/Fine_tuned_BanglaT5_CK_BN"
    models["model_chak2ban"] = AutoModelForSeq2SeqLM.from_pretrained(model_path_1).to(models["device"])
    models["tokenizer_chak2ban"] = AutoTokenizer.from_pretrained(model_path_1, use_fast=False)
    yield  # The app serves requests while paused here
    
    # This code runs AFTER the server finishes shutting down
    print("Cleaning up resources...")
    models.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

app = FastAPI(
    title = "demo",
    version = "0.0.1",
    description="Chakma Translator and Detector",
    lifespan=lifespan
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
