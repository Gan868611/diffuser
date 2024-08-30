#!/bin/bash

docker run --name diffuser-offline-rl -dit --rm --gpus all --network=host \
    --mount type=bind,source=$PWD,target=/home/code \
    --mount type=bind,source=$HOME/.d4rl,target=/root/.d4rl \
    diffuser \
