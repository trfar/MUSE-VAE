#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate SoundMap

## Set Working Directory
ROOT="/home/trfar/Documents/Advanced Machine Learning/MUSE-VAE"
cd "$ROOT"

## List of Config Files to Train
CONFIGS=(
    "model_frameworks/config.yaml"
)

## Number of runs per config
N_RUNS=1

## Temp config folder
mkdir -p temp_configs

## ============================================
## Training + Predict Loop
## ============================================

TRAIN_SCRIPT="$ROOT/model_src/Training-Pipeline.py"
PRED_SCRIPT="$ROOT/model_src/Predicting-Pipeline.py"
RECON_SCRIPT="$ROOT/model_src/Audio-Process-Pipeline.py"
LATENT_SCRIPT="$ROOT/model_src/Latent-Space-Pipeline.py"

for CONFIG in "${CONFIGS[@]}"; do

    BASENAME=$(basename "$CONFIG" .yaml)

    for ((i=1; i<=N_RUNS; i++)); do
        
        # ## Make temp config file
        TEMP_CONFIG="temp_configs/${BASENAME}_run${i}.yaml"
        cp "$CONFIG" "$TEMP_CONFIG"

        echo "============================================"
        echo "TRAINING — Config: $TEMP_CONFIG"
        echo "============================================"
        python "$TRAIN_SCRIPT" --config "$TEMP_CONFIG"

        echo
        echo "============================================"
        echo "GENERATING SPECTROGRAM PDF — Config: $TEMP_CONFIG"
        echo "============================================"
        python "$PRED_SCRIPT" --config "$TEMP_CONFIG"
        echo "Completed run $i for config: $CONFIG"
        echo

        echo
        echo "============================================"
        echo "GENERATING AUDIO RECONSTRUCTIONS — Config: $TEMP_CONFIG"
        echo "============================================"
        python "$RECON_SCRIPT" --config "$TEMP_CONFIG" --max_samples 20
        echo "Completed run $i for config: $TEMP_CONFIG"
        echo

        echo
        echo "============================================"
        echo "LATENT SPACE ANALYSIS — Config: $TEMP_CONFIG"
        echo "============================================"
        python "$LATENT_SCRIPT" --config "$TEMP_CONFIG" --split "test" --max_samples 200
        echo "Completed run $i for config: $TEMP_CONFIG"
        echo
    done
done
