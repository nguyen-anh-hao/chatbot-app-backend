from dotenv import load_dotenv
import os

load_dotenv()

# Database configuration
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/chat_app")

# Authentication
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
SECRET_KEY = os.getenv("SECRET_KEY", "your-super-secret-key-change-this-in-production")

# CORS
REDIRECT_RESPONSE = os.getenv("REDIRECT_RESPONSE", "http://localhost:3000")

# Model paths (updated for Docker)
LLAMA_MODEL_PATH = os.getenv("LLAMA_MODEL_PATH", "./Llama-3.2-1B-Instruct")
ADAPTER_PATH = os.getenv("ADAPTER_PATH", "./checkpoint-2450")
RAG_MODEL_PATH = os.getenv("RAG_MODEL_PATH", "./model_artifacts")

# File upload settings
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads/images")
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", 5 * 1024 * 1024))  # 5MB

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_DIR = os.getenv("LOG_DIR", "logs")