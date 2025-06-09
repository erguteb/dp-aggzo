#!/bin/bash

# Use a base random seed for reproducible but different random directions and DP noise across iterations:
# HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=0 DPZERO_PRIVACY_EPS=6 DP_SAMPLE_RATE=0.0625 STEP=1000 SEED=42 NUM_DIRECTION=64 RANDOM_DIRECTION_SEED=100 LR=4e-5 DPZERO_THRESHOLD=15  TASK="RTE" bash examples/dpaggzo.sh

TASK=${TASK:-SST-2}
K=${K:-512}
SEED=${SEED:-42}
BS=${BS:-64}
LR=${LR:-1e-6}
EPS=${EPS:-1e-3}
WD=${WD:-0}
STEP=${STEP:-1000}
EVAL_STEP=${EVAL_STEP:-100}
MODEL=${MODEL:-roberta-large}

DPZERO_THRESHOLD=${DPZERO_THRESHOLD:-200.0}
DPZERO_PRIVACY_EPS=${DPZERO_PRIVACY_EPS:-6.0} # -1 means non-DP, clipping threshold is then set to inf and no noise is injected
DPZERO_PRIVACY_DELTA=${DPZERO_PRIVACY_DELTA:-1e-5}
DP_SAMPLE_RATE=${DP_SAMPLE_RATE:-0.0416}
# 0.0416 for MNLI for expected BS=64
# 0.062 for SST-2 for expected BS=64
NUM_DIRECTION=${NUM_DIRECTION:-64} # number of random directions
RANDOM_DIRECTION_SEED=${RANDOM_DIRECTION_SEED:--1} # base seed for random directions and DP noise; -1 means use truly random seeds
SAMPLER_SEED=${SAMPLER_SEED:--1} # seed for batch sampling; -1 means use default seed

LOGITS=$(jq -n '{"SNLI": 3, "MNLI": 3, "trec": 6, "sst-5": 5}["'$TASK'"] // 2')

# SST-2 sst-5 SNLI MNLI trec RTE

# GR_TAG=seed$SEED-bs$BS-lr$LR-eps$EPS-wd$WD-step$STEP-evalstep$EVAL_STEP
GR_TAG=seed$SEED-lr$LR-eps$EPS-wd$WD-step$STEP-evalstep$EVAL_STEP
EXTRA_TAG=${EXTRA_TAG:-ft-}
TAG=${TAG:-k${K}-${MODEL}-dpzero-${TASK}-${EXTRA_TAG}}
echo "Grid search tag: $GR_TAG"
echo "Tag: $TAG"

TYPE=prompt GRID_TAG=$GR_TAG TAG=$TAG STEPS=$STEP TASK=$TASK SEED=$SEED MODEL=$MODEL K=$K \
    bash examples/run_fewshot_aggzo.sh \
    --per_device_train_batch_size $BS \
    --learning_rate $LR \
    --eval_steps $EVAL_STEP \
    --weight_decay $WD \
    --zero_order_eps $EPS \
    --zero_order_optim \
    --lr_scheduler_type "constant" \
    --optimizer "sgd" \
    --efficient_zero_order \
    --dpzero \
    --dpzero_clip_threshold $DPZERO_THRESHOLD \
    --dp_epsilon $DPZERO_PRIVACY_EPS \
    --dp_delta $DPZERO_PRIVACY_DELTA \
    --dp_sample_rate $DP_SAMPLE_RATE \
    --n $NUM_DIRECTION \
    --random_direction_seed $RANDOM_DIRECTION_SEED \
    --sampler_seed $SAMPLER_SEED \
    $@

