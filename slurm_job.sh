partition=cnu
code=$PROJ_CODE

gpu_type="A100"
num_gpus=1
mem=124

NUM_CPU=16
TIME=96

run_length="short"
# run_length="long"

# lbatch  -c $NUM_CPU \
#         -g $num_gpus \
#         --gputype $gpu_type \
#         -m $mem \
#         -t $TIME \
#         -a $code \
#         -q $partition \
#         -n d3pm_text8_$run_length \
#         --conda-env diffusion \
#         --cmd "python -u train.py d3pm_text8 --run_length $run_length"

# lbatch  -c $NUM_CPU \
#         -g 2 \
#         --gputype $gpu_type \
#         -m $mem \
#         -t $TIME \
#         -a $code \
#         -q $partition \
#         -n d3pm_text8_2gpu_$run_length \
#         --conda-env diffusion \
#         --cmd "python -u train.py text8_2gpu --model-type d3pm --run_length $run_length"
                                
lbatch  -c $NUM_CPU \
        -g 2 \
        --gputype $gpu_type \
        -m $mem \
        -t $TIME \
        -a $code \
        -q $partition \
        -n md4_text8_2gpu_$run_length \
        --conda-env diffusion \
        --cmd "python -u train.py text8_2gpu --model-type md4 --run_length $run_length"