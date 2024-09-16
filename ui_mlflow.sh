#!/bin/bash

# Default port
PORT=8080

# Check if an argument is provided
if [ "$#" -eq 1 ]; then
    PORT=$1
elif [ "$#" -gt 1 ]; then
    echo "Usage: $0 [port]"
    exit 1
fi

# Start the MLflow UI with the specified or default port
mlflow ui --port $PORT --backend-store-uri sqlite:///mlruns.db