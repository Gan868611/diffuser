#!/bin/bash

python scripts/train_classifier_free.py --dataset hopper-medium-v2 \
        --p_uncond 0.5 \
        --suffix p_uncond_0.5

#walker2d-medium-replay-v2
#hopper-medium-v2
