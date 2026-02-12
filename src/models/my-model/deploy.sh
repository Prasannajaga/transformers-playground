#!/bin/bash

# ================= CONFIG =================
PROJECT_ID="transformers-model-486215-a2"
REGION="us-central1"
REPO="llama-repo"
SERVICE_NAME="llama-service"
IMAGE_NAME="llama-gguf"
TAG="latest"
# ==========================================

set -e

# 1. Set Project
echo "Setting project to $PROJECT_ID..."
gcloud config set project $PROJECT_ID

# 2. Enable APIs (Cloud Build is critical here)
echo "Enabling services..."
gcloud services enable \
  artifactregistry.googleapis.com \
  run.googleapis.com \
  cloudbuild.googleapis.com

# 3. Create Repo if missing
echo "Ensuring Artifact Registry repo exists..."
gcloud artifacts repositories create $REPO \
  --repository-format=docker \
  --location=$REGION \
  --description="Llama GGUF repo" || true

# 4. BUILD IN THE CLOUD (The Critical Fix)
# This sends your Dockerfile to Google. Google builds it on a Linux server.
# This ensures the binary is 100% compatible with Cloud Run.
IMAGE_URI="$REGION-docker.pkg.dev/$PROJECT_ID/$REPO/$IMAGE_NAME:$TAG"

echo "🚀 Submitting build to Cloud Build..."
gcloud builds submit . --tag $IMAGE_URI

# 5. Deploy
echo "Deploying to Cloud Run..."
gcloud run deploy $SERVICE_NAME \
  --image $IMAGE_URI \
  --region $REGION \
  --platform managed \
  --cpu 2 \
  --memory 4Gi \
  --port 8080 \
  --concurrency 1 \
  --timeout 300 \
  --min-instances 0 \
  --max-instances 1 \
  --allow-unauthenticated

echo "✅ Deployment complete."