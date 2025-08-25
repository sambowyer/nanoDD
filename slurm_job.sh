partition=cnu
code=$PROJ_CODE

gpu_type="A100"
num_gpus=1
mem=124

NUM_CPU=16
TIME=96

# lbatch  -c $NUM_CPU \
#         -g $num_gpus \
#         --gputype $gpu_type \
#         -m $mem \
#         -t $TIME \
#         -a $code \
#         -q $partition \
#         -n d3pm_text8 \
#         --conda-env diffusion \
#         --cmd "python -u train.py d3pm_text8"

lbatch  -c $NUM_CPU \
        -g 2 \
        --gputype $gpu_type \
        -m $mem \
        -t $TIME \
        -a $code \
        -q $partition \
        -n d3pm_text8_2gpu \
        --conda-env diffusion \
        --cmd "python -u train.py d3pm_text8_2gpu"
                                