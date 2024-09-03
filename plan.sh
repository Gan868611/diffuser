#!/bin/bash

python scripts/plan_guided.py --dataset halfcheetah-medium-expert-v2 --logbase logs \
        --horizon 4 --n_diffusion_steps 5 --discount 0.99 \
        --value_loadpath 'f:values/defaults_H{horizon}_T{n_diffusion_steps}_d{discount}' \