#!/bin/bash

# ReTaMTSF Resource Consumption Benchmark Scripts
# This script runs benchmarks to measure forward time, peak RAM, and GPU hours

# Activate conda environment
source /opt/conda/etc/profile.d/conda.sh
conda activate /opt/data/private/envs/ReTaMTSF

export CUDA_VISIBLE_DEVICES=0

# Configuration
TARGET_IDS="Air Quality Index"
BATCH_SIZE=32
SEQ_LEN=96
NUM_ITERATIONS=50
OUTPUT_DIR="./benchmark_results"

# Create output directory
mkdir -p $OUTPUT_DIR

# Log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$OUTPUT_DIR/benchmark_log_${TIMESTAMP}.log"

echo "========================================" | tee -a $LOG_FILE
echo "ReTaMTSF Resource Consumption Benchmark" | tee -a $LOG_FILE
echo "========================================" | tee -a $LOG_FILE
echo "Target: $TARGET_IDS" | tee -a $LOG_FILE
echo "Batch Size: $BATCH_SIZE" | tee -a $LOG_FILE
echo "Sequence Length: $SEQ_LEN" | tee -a $LOG_FILE
echo "Iterations per config: $NUM_ITERATIONS" | tee -a $LOG_FILE
echo "Output Directory: $OUTPUT_DIR" | tee -a $LOG_FILE
echo "Log File: $LOG_FILE" | tee -a $LOG_FILE
echo "========================================" | tee -a $LOG_FILE

# =================================================================
# 1. Comparative Benchmark (Unimodal vs Multimodal)
#    Tests baseline resource consumption comparison
# =================================================================
echo "" | tee -a $LOG_FILE
echo ">>> Running Comparative Benchmark (Unimodal vs Multimodal)..." | tee -a $LOG_FILE
echo "    Configuration: H=48, K_T=3, K_n=4" | tee -a $LOG_FILE

/opt/data/private/envs/ReTaMTSF/bin/python run_benchmark_Environment.py \
    --random_seed 2025 \
    --is_training 0 \
    --root_path './dataset' \
    --TS_data_path 'Time-MMD/numerical/Environment/Environment.parquet' \
    --QT_emb_path 'Time-MMD/textual/Environment/QueryText-embedding-paraphrase-MiniLM-L6-v2-ForecastingPoint' \
    --Des_emb_path 'Time-MMD/textual/Environment/QueryText-embedding-paraphrase-MiniLM-L6-v2-Description' \
    --NewsDatabase_path 'Time-MMD/textual/Environment/NewsDatabase-embedding-paraphrase-MiniLM-L6-v2' \
    --checkpoints './checkpoints/' \
    --target_ids "$TARGET_IDS" \
    --batch_size $BATCH_SIZE \
    --num_data 1 \
    --patience 10 \
    --train_epochs 100 \
    --nperseg 48 \
    --nref 3 \
    --naggregation 3 \
    --nref_text 4 \
    --qt_ref_text 4 \
    --seq_len $SEQ_LEN \
    --pred_len 48 \
    --label_len 48 \
    --stride 1 \
    --learning_rate 1e-4 \
    --weight_decay 1e-4 \
    --dropout_rate 0.4 \
    --itr 1 \
    --num_workers 2 \
    --lradj 'type3' \
    --use_gpu True \
    --gpu 0 \
    --benchmark_type comparative \
    --num_iterations $NUM_ITERATIONS \
    --output_dir $OUTPUT_DIR 2>&1 | tee -a $LOG_FILE

# =================================================================
# 2. Sweep over Horizon Window Length (H / pred_len)
#    Both Unimodal and Multimodal
#    Note: Python script internally runs both modes, no shell loop needed
# =================================================================
echo "" | tee -a $LOG_FILE
echo ">>> Running Parameter Sweep: Horizon Window Length (H)" | tee -a $LOG_FILE
echo "    Values: [48, 96, 192, 336]" | tee -a $LOG_FILE
echo "    K_T=3, K_n=4" | tee -a $LOG_FILE

/opt/data/private/envs/ReTaMTSF/bin/python run_benchmark_Environment.py \
    --random_seed 2025 \
    --is_training 0 \
    --root_path './dataset' \
    --TS_data_path 'Time-MMD/numerical/Environment/Environment.parquet' \
    --QT_emb_path 'Time-MMD/textual/Environment/QueryText-embedding-paraphrase-MiniLM-L6-v2-ForecastingPoint' \
    --Des_emb_path 'Time-MMD/textual/Environment/QueryText-embedding-paraphrase-MiniLM-L6-v2-Description' \
    --NewsDatabase_path 'Time-MMD/textual/Environment/NewsDatabase-embedding-paraphrase-MiniLM-L6-v2' \
    --checkpoints './checkpoints/' \
    --target_ids "$TARGET_IDS" \
    --batch_size $BATCH_SIZE \
    --num_data 1 \
    --patience 10 \
    --train_epochs 100 \
    --nperseg 48 \
    --naggregation 3 \
    --nref_text 4 \
    --qt_ref_text 4 \
    --seq_len $SEQ_LEN \
    --label_len 48 \
    --stride 1 \
    --learning_rate 1e-4 \
    --weight_decay 1e-4 \
    --dropout_rate 0.4 \
    --itr 1 \
    --num_workers 2 \
    --lradj 'type3' \
    --use_gpu True \
    --gpu 0 \
    --benchmark_type sweep_H \
    --sweep_values 48 96 192 336 \
    --num_iterations $NUM_ITERATIONS \
    --output_dir $OUTPUT_DIR 2>&1 | tee -a $LOG_FILE

# =================================================================
# 3. Sweep over Time Series References (K_T / nref)
#    Both Unimodal and Multimodal
#    Note: Python script internally runs both modes, no shell loop needed
# =================================================================
echo "" | tee -a $LOG_FILE
echo ">>> Running Parameter Sweep: Time Series References (K_T)" | tee -a $LOG_FILE
echo "    Values: [1, 2, 3, 4, 5, 6, 7, 8]" | tee -a $LOG_FILE
echo "    H=48, K_n=4" | tee -a $LOG_FILE

/opt/data/private/envs/ReTaMTSF/bin/python run_benchmark_Environment.py \
    --random_seed 2025 \
    --is_training 0 \
    --root_path './dataset' \
    --TS_data_path 'Time-MMD/numerical/Environment/Environment.parquet' \
    --QT_emb_path 'Time-MMD/textual/Environment/QueryText-embedding-paraphrase-MiniLM-L6-v2-ForecastingPoint' \
    --Des_emb_path 'Time-MMD/textual/Environment/QueryText-embedding-paraphrase-MiniLM-L6-v2-Description' \
    --NewsDatabase_path 'Time-MMD/textual/Environment/NewsDatabase-embedding-paraphrase-MiniLM-L6-v2' \
    --checkpoints './checkpoints/' \
    --target_ids "$TARGET_IDS" \
    --batch_size $BATCH_SIZE \
    --num_data 1 \
    --patience 10 \
    --train_epochs 100 \
    --nperseg 48 \
    --naggregation 3 \
    --nref_text 4 \
    --qt_ref_text 4 \
    --seq_len $SEQ_LEN \
    --pred_len 48 \
    --label_len 48 \
    --stride 1 \
    --learning_rate 1e-4 \
    --weight_decay 1e-4 \
    --dropout_rate 0.4 \
    --itr 1 \
    --num_workers 2 \
    --lradj 'type3' \
    --use_gpu True \
    --gpu 0 \
    --benchmark_type sweep_KT \
    --sweep_values 1 2 3 4 5 6 7 8 \
    --num_iterations $NUM_ITERATIONS \
    --output_dir $OUTPUT_DIR 2>&1 | tee -a $LOG_FILE

# =================================================================
# 4. Sweep over Text References (K_n / nref_text)
#    Multimodal only (Unimodal doesn't use text)
#    Note: Python script internally runs only Multimodal for sweep_Kn
# =================================================================
echo "" | tee -a $LOG_FILE
echo ">>> Running Parameter Sweep: Text References (K_n)" | tee -a $LOG_FILE
echo "    Values: [1, 2, 3, 4, 5, 6, 7, 8]" | tee -a $LOG_FILE
echo "    H=48, K_T=3" | tee -a $LOG_FILE
echo "    Multimodal only (Unimodal doesn't use text)" | tee -a $LOG_FILE

/opt/data/private/envs/ReTaMTSF/bin/python run_benchmark_Environment.py \
    --random_seed 2025 \
    --is_training 0 \
    --root_path './dataset' \
    --TS_data_path 'Time-MMD/numerical/Environment/Environment.parquet' \
    --QT_emb_path 'Time-MMD/textual/Environment/QueryText-embedding-paraphrase-MiniLM-L6-v2-ForecastingPoint' \
    --Des_emb_path 'Time-MMD/textual/Environment/QueryText-embedding-paraphrase-MiniLM-L6-v2-Description' \
    --NewsDatabase_path 'Time-MMD/textual/Environment/NewsDatabase-embedding-paraphrase-MiniLM-L6-v2' \
    --checkpoints './checkpoints/' \
    --target_ids "$TARGET_IDS" \
    --batch_size $BATCH_SIZE \
    --num_data 1 \
    --patience 10 \
    --train_epochs 100 \
    --nperseg 48 \
    --nref 3 \
    --naggregation 3 \
    --seq_len $SEQ_LEN \
    --pred_len 48 \
    --label_len 48 \
    --stride 1 \
    --learning_rate 1e-4 \
    --weight_decay 1e-4 \
    --dropout_rate 0.4 \
    --itr 1 \
    --num_workers 2 \
    --lradj 'type3' \
    --use_gpu True \
    --gpu 0 \
    --benchmark_type sweep_Kn \
    --sweep_values 1 2 3 4 5 6 7 8 \
    --num_iterations $NUM_ITERATIONS \
    --output_dir $OUTPUT_DIR 2>&1 | tee -a $LOG_FILE

echo "" | tee -a $LOG_FILE
echo "========================================" | tee -a $LOG_FILE
echo "All benchmarks completed!" | tee -a $LOG_FILE
echo "Results saved to: $OUTPUT_DIR" | tee -a $LOG_FILE
echo "Log saved to: $LOG_FILE" | tee -a $LOG_FILE
echo "========================================" | tee -a $LOG_FILE
