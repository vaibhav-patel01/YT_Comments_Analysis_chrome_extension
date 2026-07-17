#!/bin/bash
# Log everything to start_docker.log
exec > /home/ubuntu/start_docker.log 2>&1

echo "Logging in to ECR..."
aws ecr get-login-password --region eu-north-1 | sudo docker login --username AWS --password-stdin 532918215945.dkr.ecr.eu-north-1.amazonaws.com

echo "Pulling Docker image..."
sudo docker pull 532918215945.dkr.ecr.eu-north-1.amazonaws.com/vaibhavpatel01project1:latest

echo "Checking for existing container..."
if [ "$(sudo docker ps -q -f name=yt-comments-api)" ]; then
    echo "Stopping existing container..."
    sudo docker stop yt-comments-api
fi

if [ "$(sudo docker ps -aq -f name=yt-comments-api)" ]; then
    echo "Removing existing container..."
    sudo docker rm yt-comments-api
fi

echo "Starting new container..."
sudo docker run -d \
  -p 80:8000 \
  -e DAGSHUB_PAT=e819601686b8690f7deb99aa6e31ef22ebf17655 \
  --name yt-comments-api \
  532918215945.dkr.ecr.eu-north-1.amazonaws.com/vaibhavpatel01project1:latest

echo "Container started successfully."