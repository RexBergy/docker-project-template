#!/bin/bash

# Define the image tag
IMAGE_TAG="ift6758/serving:latest"

# Build the Docker image
docker build -t $IMAGE_TAG -f Dockerfile.serving .