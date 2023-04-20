#!/bin/sh
MODEL_BASE_PATH="models"

MODEL_NAME="BERT"

SAVED_MODEL_PATH="/c/Users/Moses/Desktop/NiaAI/models/BERT"

docker run -p 8501:8501 \
  --name "tfserving_${MODEL_NAME}" \
  --mount type=bind,source="${SAVED_MODEL_PATH}",target=/${MODEL_BASE_PATH}/${MODEL_NAME} \
  -e MODEL_NAME=${MODEL_NAME} -t tensorflow/serving
