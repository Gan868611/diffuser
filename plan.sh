#!/bin/bash

python scripts/plan_guided.py --dataset hopper-medium-v2 --logbase logs \
        --horizon 4 --n_diffusion_steps 20 --discount 0.99 \
        --value_loadpath /home/code/logs/hopper-medium-v2/values/20240903_163700_defaults_H32_T20_d0.99 \
        --diffusion_loadpath /home/code/logs/hopper-medium-v2/diffusion/20240903_162200_defaults_H32_T20 \


#hopper-medium-v2
# halfcheetah-medium-expert-v2


# python scripts/plan_guided.py --dataset walker2d-medium-replay-v2 --logbase logs \
#         --horizon 4 --n_diffusion_steps 20 --discount 0.99 \
#         --value_loadpath /home/code/logs/walker2d-medium-replay-v2/values/20240903_163734_defaults_H32_T20_d0.99 \
#         --diffusion_loadpath /home/code/logs/walker2d-medium-replay-v2/diffusion/20240903_161955_defaults_H32_T20 \

