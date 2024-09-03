#!/bin/bash

docker run --name diffuser-offline-rl -dit --rm --gpus all --network=host \
    -w /home/code \
    --mount type=bind,source=$PWD,target=/home/code \
    --mount type=bind,source=$HOME/.d4rl,target=/root/.d4rl \
    diffuser \
    
docker exec -it diffuser-offline-rl /bin/bash -c "echo 'export MLFLOW_TRACKING_URI=sqlite:///mlruns.db' >> /root/.bashrc"
