# HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=0 MODEL=facebook/opt-1.3b TASK=SST2 MODE=ft LR=5e-6 EPS=1e-3 DP_SAMPLE_RATE=0.064 DP_EPS=2.0 STEPS=1000 N=64 DP_CLIP=25 bash examples/dpaggzo.sh

MODEL=${MODEL:-facebook/opt-1.3b}
MODEL_NAME=(${MODEL//\// })
MODEL_NAME="${MODEL_NAME[-1]}"

LR=${LR:-1e-5}
EPS=${EPS:-1e-3}
SEED=${SEED:-0}
TRAIN=${TRAIN:-1000}
DEV=${DEV:-500}
EVAL=${EVAL:-1000}
STEPS=${STEPS:-20000}
EVAL_STEPS=${EVAL_STEPS:-4000}
DP_EPS=${DP_EPS:-6.0}
DP_CLIP=${DP_CLIP:-10.0}

# poisson sample rate
DP_SAMPLE_RATE=${DP_SAMPLE_RATE:-0.008}
# number of directions
N=${N:-10}

MODE=${MODE:-ft}
EXTRA_ARGS=""
if [ "$MODE" == "prefix" ]; then
    EXTRA_ARGS="--prefix_tuning --num_prefix 5 --no_reparam --prefix_init_by_real_act"
elif [ "$MODE" == "lora" ]; then
    EXTRA_ARGS="--lora"
fi
TAG=dpzero-$MODE-$STEPS-$DP_SAMPLE_RATE-$LR-$EPS-$SEED-$DP_EPS-$DP_CLIP

TASK_ARGS=""
case $TASK in
    # For Copa, ReCoRD, SQuAD, DROP, we set --train_as_classification False; for others, set this flag to True
    CB) # It has <1000 training examples. Only use 100 for dev
        DEV=100
        ;;
    Copa) # It has <1000 training examples. Only use 100 for dev
        DEV=100
        TASK_ARGS="--train_as_classification False"
        ;;
    ReCoRD) 
        TASK_ARGS="--train_as_classification False"
        ;;
    DROP) 
        TASK_ARGS="--train_as_classification False"
        ;;
    SQuAD)
        TASK_ARGS="--train_as_classification False"
        ;;
esac

echo $TAG
echo $TASK
echo "Poisson sample rate: $DP_SAMPLE_RATE"
echo "Expected BS $(echo "$DP_SAMPLE_RATE * 1000" | bc)" 
echo "LR: $LR"
echo "EPS: $EPS"
echo "SEED: $SEED"
echo "TRAIN/EVAL STEPS: $STEPS/$EVAL_STEPS"
echo "MODE: $MODE"
echo "DP_EPS: $DP_EPS"
echo "DP_CLIP: $DP_CLIP"
echo "number of directions: $N"
# echo "Extra args: $EXTRA_ARGS $TASK_ARGS"

python run_dpaggzo.py \
    --model_name $MODEL \
    --task_name $TASK \
    --output_dir result/$TASK-${MODEL_NAME}-$TAG --tag $TAG --train_set_seed $SEED --num_train $TRAIN --num_dev $DEV --num_eval $EVAL --logging_steps 10 \
    --max_steps $STEPS \
    --trainer zo --load_float16 \
    --learning_rate $LR --zo_eps $EPS --lr_scheduler_type "constant" \
    --load_best_model_at_end --evaluation_strategy steps --save_strategy steps --save_total_limit 1 \
    --eval_steps $EVAL_STEPS --save_steps $EVAL_STEPS \
    --train_as_classification \
    --dpzero \
    --dpzero_clip_threshold $DP_CLIP \
    --dp_epsilon $DP_EPS \
    --dp_delta 1e-5 \
    --dp_sample_rate $DP_SAMPLE_RATE \
    --n $N \
    $EXTRA_ARGS \
    $TASK_ARGS \
    "$@"

echo $TAG
echo $MODEL_NAME
echo $TASK
echo "Poisson sample rate: $DP_SAMPLE_RATE"
echo "Expected BS $(echo "$DP_SAMPLE_RATE * 1000" | bc)" 
echo "LR: $LR"
echo "EPS: $EPS"
echo "SEED: $SEED"
echo "TRAIN/EVAL STEPS: $STEPS/$EVAL_STEPS"
echo "MODE: $MODE"
echo "DP_EPS: $DP_EPS"
echo "DP_CLIP: $DP_CLIP"
echo "number of directions: $N"
# echo "Extra args: $EXTRA_ARGS $TASK_ARGS"

