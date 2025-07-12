# Chatbot Backend API

A comprehensive chatbot backend with RAG (Retrieval-Augmented Generation) capabilities, built with FastAPI and powered by Llama models.

## 🚀 Quick Start with Docker

### Prerequisites
- Docker and Docker Compose installed
- NVIDIA Docker runtime (for GPU support)
- At least 8GB of available RAM
- NVIDIA GPU with CUDA support (recommended)

### 1. Clone and Setup
```bash
git clone <your-repo>
cd chatbot-app-backend

# Copy environment template
cp .env.example .env

# Edit .env with your configuration
nano .env
```

### 2. Prepare Models
Ensure you have the required model files:
- `./Llama-3.2-1B-Instruct/` - Base Llama model
- `./checkpoint-4990/` - Fine-tuned adapter
- `./model_artifacts/` - RAG model artifacts

### 3. Build and Run
```bash
# Build and start all services
docker-compose up --build

# Or run in detached mode
docker-compose up -d --build

# View logs
docker-compose logs -f chatbot-api
```

### 4. Access the API
- API: http://localhost:8000
- Health Check: http://localhost:8000/health
- API Docs: http://localhost:8000/docs
- MongoDB: localhost:27017

## 🛠️ Development Setup (without Docker)

### Using Conda
```bash
# Create environment from file
conda env create -f environment.yml

# Activate
conda activate chat-env

# Install additional requirements
pip install -r requirements.txt

# Run
uvicorn main:app --host 0.0.0.0 --port 8000
```

## 📦 Docker Commands

```bash
# Build only the API image
docker build -t chatbot-api .

# Start only database
docker-compose up mongo

# Restart API service
docker-compose restart chatbot-api

# View service status
docker-compose ps

# Stop all services
docker-compose down

# Remove volumes (careful - this deletes data!)
docker-compose down -v
```

## 🔧 Configuration

### Environment Variables
- `GOOGLE_CLIENT_ID` - Google OAuth client ID
- `GOOGLE_CLIENT_SECRET` - Google OAuth secret
- `MONGO_URI` - MongoDB connection string
- `SECRET_KEY` - JWT secret key
- `REDIRECT_RESPONSE` - Frontend URL for CORS

### Model Configuration
- `LLAMA_MODEL_PATH` - Path to base Llama model
- `ADAPTER_PATH` - Path to fine-tuned adapter
- `RAG_MODEL_PATH` - Path to RAG model artifacts

## 📁 Project Structure
```
chatbot-app-backend/
├── auth/              # Authentication modules
├── routes/            # API route handlers
├── database/          # Database connections
├── models/            # Pydantic models
├── uploads/           # File uploads
├── logs/              # Application logs
├── model_artifacts/   # RAG model files
├── Llama-3.2-1B-Instruct/  # Base model
├── checkpoint-4990/        # Fine-tuned adapter
├── Dockerfile              # Docker configuration
├── docker-compose.yml # Multi-service setup
├── requirements.txt   # Python dependencies
└── main.py            # FastAPI application
```

## 🔥 Production Deployment

### Using Docker Swarm
```bash
# Initialize swarm
docker swarm init

# Deploy stack
docker stack deploy -c docker-compose.yml chatbot

# Scale services
docker service scale chatbot_chatbot-api=3
```

### Using Kubernetes
```bash
# Convert docker-compose to k8s (using kompose)
kompose convert

# Apply to cluster
kubectl apply -f .
```

## 🚨 Troubleshooting

### Common Issues
1. **CUDA out of memory**: Reduce batch size or use CPU mode
2. **Model not found**: Ensure model files are in correct paths
3. **Permission denied**: Check file permissions for uploads directory
4. **MongoDB connection failed**: Verify MongoDB service is running

### Logs
```bash
# View all logs
docker-compose logs

# Follow API logs
docker-compose logs -f chatbot-api

# View MongoDB logs
docker-compose logs mongo
```

## 🔒 Security Notes
- Change `SECRET_KEY` in production
- Use HTTPS in production
- Configure proper CORS origins
- Keep Google OAuth credentials secure
- Regular security updates for dependencies

## 📊 Monitoring
- Health check: `GET /health`
- Metrics endpoint: `GET /metrics` (if implemented)
- Container stats: `docker stats`