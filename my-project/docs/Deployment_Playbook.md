Deployment_Playbook.md - Release Procedures
Overview
This playbook provides step-by-step procedures for deploying the Ad-Pod Stitching Server Optimization application.
Local Deployment

Backend: uvicorn backend.app:app --host 0.0.0.0 --port 8000.
Frontend: streamlit run frontend/app.py.

Docker Deployment

Build: docker build -t ad-pod-optimizer ..
Compose: docker-compose up -d (exposes 8000, 8501).
Dockerfile: FROM python:3.10-slim, COPY ., pip install -r requirements.txt, CMD for uvicorn.

Production Deployment

Platform: Heroku/AWS/EC2.
Steps:

Build Docker image, push to registry.
Deploy containers (e.g., ECS/Kubernetes).
Add NGINX reverse proxy for SSL.
Env vars: API_URL for frontend.


Monitoring: Add health checks (/health endpoint), logs to CloudWatch.

Rollback

Tag releases, docker-compose down && up with previous tag.

This playbook ensures smooth, error-free releases.