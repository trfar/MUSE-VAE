#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate SoundMap

## Set Working Directory
ROOT="/home/trfar/Documents/Stats/MUSE-VAE"
cd "$ROOT"

## List of Config Files to Train
CONFIGS=(
    "model_frameworks/config.yaml"
)

## Number of runs per config
N_RUNS=5

## Temp config folder
mkdir -p temp_configs

## ============================================
## Training + Predict Loop
## ============================================

TRAIN_SCRIPT="$ROOT/model_src/Training-Pipeline.py"
PRED_SCRIPT="$ROOT/model_src/Predicting-Pipeline.py"

for CONFIG in "${CONFIGS[@]}"; do

    BASENAME=$(basename "$CONFIG" .yaml)

    for ((i=1; i<=N_RUNS; i++)); do
        
        ## Make temp config file
        TEMP_CONFIG="temp_configs/${BASENAME}_run${i}.yaml"
        cp "$CONFIG" "$TEMP_CONFIG"

        echo "============================================"
        echo "TRAINING — Config: $TEMP_CONFIG"
        echo "============================================"

        python "$TRAIN_SCRIPT" --config "$TEMP_CONFIG"

        echo
        echo "============================================"
        echo "PREDICTING — Config: $TEMP_CONFIG"
        echo "============================================"

        python "$PRED_SCRIPT" --config "$TEMP_CONFIG"

        echo "Completed run $i for config: $CONFIG"
        echo
    done
done
