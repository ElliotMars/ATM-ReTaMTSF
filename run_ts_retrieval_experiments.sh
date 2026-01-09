#!/usr/bin/env bash
set -euo pipefail

# Environment variables
export PATH=~/git-2.41.0/git:$PATH
export PATH=/opt/data/private/ollama/bin:$PATH
export PATH=/opt/conda/bin:$PATH
export OLLAMA_MODELS=/opt/data/private/ollama-models/
export OLLAMA_HOST=0.0.0.0:11434
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0,1,2,3

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/opt/conda/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
  eval "$__conda_setup"
else
  if [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    . "/opt/conda/etc/profile.d/conda.sh"
  else
    export PATH="/opt/conda/bin:$PATH"
  fi
fi
unset __conda_setup
# <<< conda initialize <<<

conda activate /opt/data/private/envs/ReTaMTSF

PYTHON=/opt/data/private/envs/ReTaMTSF/bin/python
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RESULTS_DIR="${ROOT_DIR}/ts_retrieval"
mkdir -p "${RESULTS_DIR}"
SUMMARY_CSV="${RESULTS_DIR}/summary.csv"
echo "Run timestamp: $(date +%Y%m%d_%H%M%S)" > "${RESULTS_DIR}/run_info.txt"
echo "dataset,ts_retrieval,pred_len,mse,mae,rse,result_path" > "${SUMMARY_CSV}"

DATASETS=("Energy" "Climate" "Health_US" "Environment")
TS_RETRIEVALS=("lagged_corr" "partial_corr" "granger")

run_and_collect() {
  local dataset="$1"
  local ts_retrieval="$2"
  local pred_len="$3"
  shift 3
  local cmd=("$@")

  local log_file="${RESULTS_DIR}/${dataset}_${ts_retrieval}_H${pred_len}.log"
  echo "Running ${dataset} | ts_retrieval=${ts_retrieval} | pred_len=${pred_len}"
  "${cmd[@]}" > "${log_file}" 2>&1

  local latest_dir
  latest_dir="$(ls -td ./test_results/* | head -n 1)"
  local result_file="${latest_dir}/result.txt"
  local out_copy="${RESULTS_DIR}/${dataset}_${ts_retrieval}_H${pred_len}_result.txt"
  cp "${result_file}" "${out_copy}"

  local line
  line="$(rg -m1 "mse:" "${out_copy}" || true)"
  local mse=""; local mae=""; local rse=""
  if [[ -n "${line}" ]]; then
    read -r mse mae rse <<<"$(echo "${line}" | awk -F'[,:]' '{print $2,$4,$6}')"
  fi
  echo "${dataset},${ts_retrieval},${pred_len},${mse},${mae},${rse},${out_copy}" >> "${SUMMARY_CSV}"
  echo "Result: ${line}"
}

for dataset in "${DATASETS[@]}"; do
  for ts_retrieval in "${TS_RETRIEVALS[@]}"; do
    random_seed=2025
    case "${dataset}" in
      "Energy")
        for pred_len in 12 24 36 48; do
          run_and_collect "${dataset}" "${ts_retrieval}" "${pred_len}" \
              "${PYTHON}" run_ReTaMTSF_Energy.py \
                --random_seed "${random_seed}" \
                --is_training 1 \
                --ablation 0 \
                --ablation_syn 0 \
                --root_path './dataset' \
                --TS_data_path 'Time-MMD/numerical/Energy/Energy.parquet' \
                --QT_emb_path 'Time-MMD/textual/Energy/QueryText-embedding-paraphrase-MiniLM-L6-v2-ForecastingPoint' \
                --Des_emb_path 'Time-MMD/textual/Energy/QueryText-embedding-paraphrase-MiniLM-L6-v2-Description' \
                --NewsDatabase_path 'Time-MMD/textual/Energy/NewsDatabase-embedding-paraphrase-MiniLM-L6-v2' \
                --checkpoints './checkpoints/' \
                --target_ids "Gasoline Prices" \
                --batch_size 32 \
                --num_data 1 \
                --patience 10 \
                --train_epochs 100 \
                --nperseg 18 \
                --nref 5 \
                --naggregation 3 \
                --nref_text 6 \
                --qt_ref_text 6 \
                --seq_len 36 \
                --pred_len "${pred_len}" \
                --label_len 18 \
                --stride 1 \
                --learning_rate 1e-4 \
                --weight_decay 1e-3 \
                --dropout_rate 0.63 \
                --itr 1 \
                --num_workers 2 \
                --pct_start 0.3 \
                --lradj 'type3' \
                --use_gpu True \
                --devices '0,1,2,3' \
                --gpu 0 \
                --use_multi_gpu \
                --text_mode "retrieval" \
                --ts_retrieval "${ts_retrieval}" \
                --max_lag 12 \
                --granger_lag 3
        done
        ;;
      "Climate")
        for pred_len in 6 8 10 12; do
          run_and_collect "${dataset}" "${ts_retrieval}" "${pred_len}" \
              "${PYTHON}" run_ReTaMTSF_Climate.py \
                --random_seed "${random_seed}" \
                --is_training 1 \
                --ablation 0 \
                --ablation_syn 0 \
                --root_path './dataset' \
                --TS_data_path 'Time-MMD/numerical/Climate/Climate.parquet' \
                --QT_emb_path 'Time-MMD/textual/Climate/QueryText-embedding-paraphrase-MiniLM-L6-v2-ForecastingPoint' \
                --Des_emb_path 'Time-MMD/textual/Climate/QueryText-embedding-paraphrase-MiniLM-L6-v2-Description' \
                --NewsDatabase_path 'Time-MMD/textual/Climate/NewsDatabase-embedding-paraphrase-MiniLM-L6-v2' \
                --checkpoints './checkpoints/' \
                --target_ids "Drought Level" \
                --batch_size 32 \
                --num_data 1 \
                --patience 10 \
                --train_epochs 100 \
                --nperseg 4 \
                --nref 5 \
                --naggregation 3 \
                --nref_text 3 \
                --qt_ref_text 3 \
                --seq_len 8 \
                --pred_len "${pred_len}" \
                --label_len 4 \
                --stride 1 \
                --learning_rate 1e-4 \
                --weight_decay 1e-5 \
                --dropout_rate 0.5 \
                --itr 1 \
                --num_workers 2 \
                --pct_start 0.3 \
                --lradj 'type3' \
                --use_gpu True \
                --devices '0,1,2,3' \
                --gpu 0 \
                --use_multi_gpu \
                --text_mode "retrieval" \
                --ts_retrieval "${ts_retrieval}" \
                --max_lag 12 \
                --granger_lag 3
        done
        ;;
      "Health_US")
        for pred_len in 12 24 36 48; do
          run_and_collect "${dataset}" "${ts_retrieval}" "${pred_len}" \
              "${PYTHON}" run_ReTaMTSF_Health_US.py \
                --random_seed "${random_seed}" \
                --is_training 1 \
                --ablation 0 \
                --ablation_syn 0 \
                --root_path './dataset' \
                --TS_data_path 'Time-MMD/numerical/Health_US/Health_US.parquet' \
                --QT_emb_path 'Time-MMD/textual/Health_US/QueryText-embedding-paraphrase-MiniLM-L6-v2-ForecastingPoint' \
                --Des_emb_path 'Time-MMD/textual/Health_US/QueryText-embedding-paraphrase-MiniLM-L6-v2-Description' \
                --NewsDatabase_path 'Time-MMD/textual/Health_US/NewsDatabase-embedding-paraphrase-MiniLM-L6-v2' \
                --checkpoints './checkpoints/' \
                --target_ids "Influenza Patients Proportion" \
                --batch_size 32 \
                --num_data 1 \
                --patience 10 \
                --train_epochs 100 \
                --nperseg 18 \
                --nref 5 \
                --naggregation 3 \
                --nref_text 11 \
                --qt_ref_text 11 \
                --seq_len 36 \
                --pred_len "${pred_len}" \
                --label_len 18 \
                --stride 1 \
                --learning_rate 1e-4 \
                --weight_decay 1e-4 \
                --dropout_rate 0.2 \
                --itr 1 \
                --num_workers 2 \
                --pct_start 0.3 \
                --lradj 'type3' \
                --use_gpu True \
                --devices '0,1,2,3' \
                --gpu 0 \
                --use_multi_gpu \
                --text_mode "retrieval" \
                --ts_retrieval "${ts_retrieval}" \
                --max_lag 12 \
                --granger_lag 3
        done
        ;;
      "Environment")
        for pred_len in 48 96 192 336; do
          run_and_collect "${dataset}" "${ts_retrieval}" "${pred_len}" \
              "${PYTHON}" run_ReTaMTSF_Environment.py \
                --random_seed "${random_seed}" \
                --is_training 1 \
                --ablation 0 \
                --ablation_syn 0 \
                --root_path './dataset' \
                --TS_data_path 'Time-MMD/numerical/Environment/Environment.parquet' \
                --QT_emb_path 'Time-MMD/textual/Environment/QueryText-embedding-paraphrase-MiniLM-L6-v2-ForecastingPoint' \
                --Des_emb_path 'Time-MMD/textual/Environment/QueryText-embedding-paraphrase-MiniLM-L6-v2-Description' \
                --NewsDatabase_path 'Time-MMD/textual/Environment/NewsDatabase-embedding-paraphrase-MiniLM-L6-v2' \
                --checkpoints './checkpoints/' \
                --target_ids "Air Quality Index" \
                --batch_size 32 \
                --num_data 1 \
                --patience 10 \
                --train_epochs 100 \
                --nperseg 48 \
                --nref 2 \
                --naggregation 3 \
                --nref_text 4 \
                --qt_ref_text 4 \
                --seq_len 96 \
                --pred_len "${pred_len}" \
                --label_len 48 \
                --stride 1 \
                --learning_rate 1e-4 \
                --weight_decay 1e-4 \
                --dropout_rate 0.4 \
                --itr 1 \
                --num_workers 2 \
                --pct_start 0.3 \
                --lradj 'type3' \
                --use_gpu True \
                --devices '0,1,2,3' \
                --gpu 0 \
                --use_multi_gpu \
                --text_mode "retrieval" \
                --ts_retrieval "${ts_retrieval}" \
                --max_lag 12 \
                --granger_lag 3
        done
        ;;
      *)
        echo "Unknown dataset: ${dataset}"
        exit 1
        ;;
    esac
  done
done

echo "Done. Summary saved to ${SUMMARY_CSV}"
