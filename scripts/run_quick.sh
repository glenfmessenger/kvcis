#!/bin/bash
#
# KVCIS Quick Validation Script
#
# Fast validation of core results (~15 minutes on A100)
# Uses fewer prompts and samples for speed
#
# Usage:
#   ./run_quick.sh
#

set -e

echo "========================================"
echo "KVCIS Quick Validation"
echo "========================================"
echo ""

# Configuration
MODEL="meta-llama/Llama-3.1-8B-Instruct"
EXTRACTION_LAYER=12
N_PROMPTS=50
N_TEXTS=10
OUTPUT_DIR="../output_quick_$(date +%Y%m%d_%H%M%S)"

mkdir -p "$OUTPUT_DIR"

cd code

echo "[1/4] Testing activation extraction..."
python step1_single_prompt.py \
    --model "$MODEL" \
    --extraction-layer "$EXTRACTION_LAYER"

echo ""
echo "[2/4] Collecting training data (${N_PROMPTS} prompts)..."
python step2_collect_data.py \
    --model "$MODEL" \
    --extraction-layer "$EXTRACTION_LAYER" \
    --n-prompts "$N_PROMPTS" \
    --output-dir "$OUTPUT_DIR/data"

echo ""
echo "[3/4] Training importance probe..."
python step3_train_probe.py \
    --data-dir "$OUTPUT_DIR/data" \
    --output-dir "$OUTPUT_DIR/probe"

echo ""
echo "[4/4] Evaluating compression..."
python step4_compression_eval.py \
    --model "$MODEL" \
    --probe-path "$OUTPUT_DIR/probe/regression" \
    --extraction-layer "$EXTRACTION_LAYER" \
    --n-texts "$N_TEXTS" \
    --output-dir "$OUTPUT_DIR/results"

echo ""
echo "========================================"
echo "QUICK VALIDATION COMPLETE"
echo "========================================"
echo "Results saved to: $OUTPUT_DIR"
