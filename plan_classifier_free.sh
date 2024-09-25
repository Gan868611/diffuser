#!/bin/bash


export CUDA_VISIBLE_DEVICES=0

seeds=(1 2 3)

function process_item {
    num_step=$1
    update_epoch=$2
    clip_coef=$3
    num_minibatch=$4
    policy=$5

    for seed in "${seeds[@]}"
    do
        # export CUDA_VISIBLE_DEVICES=$gpu
        nohup python3 ./script/train.py num_steps=$num_step seed=$seed policy=$policy num_minibatches=$num_minibatch clip_coef=$clip_coef update_epochs=$update_epoch file_name_comment="cc_${clip_coef}__num_mb_${num_minibatch}__up_epoch_${update_epoch}__num_step_${num_step}" >> "./nohup_out/${policy}__seed_${seed}__num_mb_${num_minibatch}__cc_${clip_coef}__up_epoch_${update_epoch}__num_step_${num_step}.out"  &
        sleep 2
    done

}



# Define the array of guidance weights
# guidance_weights=(2.0 5.0 10.0 15.0 20.0)
guidance_weights=(0.0 30.0 40.0 50.0)

# Loop through each guidance weight and run the command
for weight in "${guidance_weights[@]}"; do
    nohup python scripts/plan_classifier_free.py --dataset hopper-medium-v2 --logbase logs \
        --horizon 32 --n_diffusion_steps 20 --discount 0.99 \
        --diffusion_loadpath /home/code/logs/hopper-medium-v2/classifier_free/20240924_150537_defaults_H32_T20_d0.99_p_uncond_0.5 \
        --guidance_weight $weight \
        --p_uncond 0.5 \
        --suffix "p_uncond_0.5_gweight_$weight" >> "./nohup_out/p_uncond_0_${weight}.out"  &

        sleep 3
done

wait

#hopper-medium-v2
# halfcheetah-medium-expert-v2


# python scripts/plan_guided.py --dataset walker2d-medium-replay-v2 --logbase logs \
#         --horizon 4 --n_diffusion_steps 20 --discount 0.99 \
#         --value_loadpath /home/code/logs/walker2d-medium-replay-v2/values/20240903_163734_defaults_H32_T20_d0.99 \
#         --diffusion_loadpath /home/code/logs/walker2d-medium-replay-v2/diffusion/20240903_161955_defaults_H32_T20 \

