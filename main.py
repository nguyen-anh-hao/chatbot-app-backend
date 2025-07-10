from fastapi import FastAPI
from starlette.middleware.sessions import SessionMiddleware
from fastapi.middleware.cors import CORSMiddleware
import logging
import os

from auth.google_auth import google_router
from routes.chat import chat_router
from routes.history import history_router
from routes.image import image_router
import config

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Chatbot API",
    description="A comprehensive chatbot API with RAG and LLM capabilities",
    version="1.0.0"
)

app.add_middleware(
    SessionMiddleware,
    secret_key=config.SECRET_KEY,
    same_site="none",
    https_only=True,  # Set to True in production with HTTPS
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        config.REDIRECT_RESPONSE,
        "http://localhost:5173",  # Development frontend
        "https://nguyenanhhao.site"  # Production frontend
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint for Docker"""
    return {
        "status": "healthy",
        "message": "Chatbot API is running",
        "version": "1.0.0"
    }

# Include routers
app.include_router(google_router)
app.include_router(chat_router)
app.include_router(history_router)
app.include_router(image_router)

@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    logger.info("Starting Chatbot API...")
    
    # Check GPU availability
    import torch
    if torch.cuda.is_available():
        logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA version: {torch.version.cuda}")
    else:
        logger.warning("No GPU detected! Running in CPU mode (slower)")
    
    # Create necessary directories
    os.makedirs(config.UPLOAD_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)
    os.makedirs(config.RAG_MODEL_PATH, exist_ok=True)
    
    logger.info("Chatbot API started successfully!")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Chatbot API...")
