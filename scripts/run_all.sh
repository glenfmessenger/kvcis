#!/bin/bash
#
# KVCIS Full Reproduction Script
#
# Reproduces all results from the paper:
# - Table 1: Baseline comparison
# - Table 2: Downstream tasks
# - Table 3: Long-context scaling
# - Table 4: Cross-model validation
#
# Requirements:
# - NVIDIA GPU with 24GB+ VRAM (A100 recommended)
# - ~2 hours runtime
# - Hugging Face access to Llama-3.1-8B-Instruct
#
# Usage:
#   ./run_all.sh
#

set -e

echo "========================================"
echo "KVCIS Full Reproduction"
echo "========================================"
echo ""

# Configuration
MODEL="meta-llama/Llama-3.1-8B-Instruct"
EXTRACTION_LAYER=12
N_PROMPTS=500
N_TEXTS=50
OUTPUT_DIR="../output_$(date +%Y%m%d_%H%M%S)"

mkdir -p "$OUTPUT_DIR"

cd code

echo "[1/5] Testing activation extraction..."
python step1_single_prompt.py \
    --model "$MODEL" \
    --extraction-layer "$EXTRACTION_LAYER"

echo ""
echo "[2/5] Collecting training data (${N_PROMPTS} prompts)..."
python step2_collect_data.py \
    --model "$MODEL" \
    --extraction-layer "$EXTRACTION_LAYER" \
    --n-prompts "$N_PROMPTS" \
    --output-dir "$OUTPUT_DIR/data"

echo ""
echo "[3/5] Training importance probe..."
python step3_train_probe.py \
    --data-dir "$OUTPUT_DIR/data" \
    --output-dir "$OUTPUT_DIR/probe"

echo ""
echo "[4/5] Evaluating compression (Table 1 & 2)..."
python step4_compression_eval.py \
    --model "$MODEL" \
    --probe-path "$OUTPUT_DIR/probe/regression" \
    --extraction-layer "$EXTRACTION_LAYER" \
    --n-texts "$N_TEXTS" \
    --output-dir "$OUTPUT_DIR/results"

echo ""
echo "[5/5] Long-context evaluation (Table 3)..."
python longctx_eval.py \
    --model "$MODEL" \
    --probe-path "$OUTPUT_DIR/probe/regression" \
    --extraction-layer "$EXTRACTION_LAYER" \
    --context-lengths 512 1024 2048 \
    --n-texts 20 \
    --skip-h2o \
    --output-dir "$OUTPUT_DIR/longctx"

echo ""
echo "========================================"
echo "REPRODUCTION COMPLETE"
echo "========================================"
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Output files:"
ls -la "$OUTPUT_DIR"
