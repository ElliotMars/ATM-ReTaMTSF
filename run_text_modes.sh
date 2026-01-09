#!/usr/bin/env bash
set -euo pipefail

# conda setup
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
conda activate /opt/data/private/envs/ReTaMTSF

export CUDA_VISIBLE_DEVICES=0,1,2

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

results_dir="${ROOT_DIR}/text_mode"
mkdir -p "${results_dir}"
summary_csv="${results_dir}/summary.csv"
echo "Run timestamp: $(date +%Y%m%d_%H%M%S)" > "${results_dir}/run_info.txt"
echo "dataset,mode,pred_len,mse,mae,rse,result_path" > "${summary_csv}"

run_and_collect() {
  local dataset="$1"
  local mode="$2"
  local pred_len="$3"
  shift 3
  local cmd=("$@")

  local log_file="${results_dir}/${dataset}_${mode}_H${pred_len}.log"
  echo "Running ${dataset} | mode=${mode} | pred_len=${pred_len}"
  if ! "${cmd[@]}" > "${log_file}" 2>&1; then
    echo "Command failed. Last 50 lines of log: ${log_file}"
    tail -n 50 "${log_file}"
    return 1
  fi

  local latest_dir
  if [[ ! -d "${ROOT_DIR}/test_results" ]]; then
    echo "Missing test_results directory at ${ROOT_DIR}/test_results"
    echo "Last 50 lines of log: ${log_file}"
    tail -n 50 "${log_file}"
    return 1
  fi
  latest_dir="$(ls -td "${ROOT_DIR}"/test_results/* 2>/dev/null | head -n 1 || true)"
  if [[ -z "${latest_dir}" ]]; then
    echo "No test_results subdirectory found under ${ROOT_DIR}/test_results"
    echo "Last 50 lines of log: ${log_file}"
    tail -n 50 "${log_file}"
    return 1
  fi
  local result_file="${latest_dir}/result.txt"
  local out_copy="${results_dir}/${dataset}_${mode}_H${pred_len}_result.txt"
  if [[ ! -f "${result_file}" ]]; then
    echo "Missing result file: ${result_file}"
    echo "Last 50 lines of log: ${log_file}"
    tail -n 50 "${log_file}"
    return 1
  fi
  cp "${result_file}" "${out_copy}"

  local line
  line="$(rg -m1 "mse:" "${out_copy}" || true)"
  local mse=""; local mae=""; local rse=""
  if [[ -n "${line}" ]]; then
    read -r mse mae rse <<<"$(echo "${line}" | awk -F'[,:]' '{print $2,$4,$6}')"
  fi
  echo "${dataset},${mode},${pred_len},${mse},${mae},${rse},${out_copy}" >> "${summary_csv}"
  echo "Result: ${line}"
}

# Energy settings
for mode in aligned random; do
  for pred_len in 12 24 36 48; do
    run_and_collect "Energy" "${mode}" "${pred_len}" \
      python run_ReTaMTSF_Energy.py \
        --random_seed 2025 \
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
        --devices '0,1,2' \
        --gpu 0 \
        --use_multi_gpu \
        --text_mode "${mode}"
  done
done

# Climate settings
for mode in aligned random; do
  for pred_len in 6 8 10 12; do
    run_and_collect "Climate" "${mode}" "${pred_len}" \
      python run_ReTaMTSF_Climate.py \
        --random_seed 2025 \
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
        --devices '0,1,2' \
        --gpu 0 \
        --use_multi_gpu \
        --text_mode "${mode}"
  done
done

# Health_US settings
for mode in aligned random; do
  for pred_len in 12 24 36 48; do
    run_and_collect "Health_US" "${mode}" "${pred_len}" \
      python run_ReTaMTSF_Health_US.py \
        --random_seed 2025 \
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
        --devices '0,1,2' \
        --gpu 0 \
        --use_multi_gpu \
        --text_mode "${mode}"
  done
done

# Environment settings
for mode in aligned random; do
  for pred_len in 48 96 192 336; do
    run_and_collect "Environment" "${mode}" "${pred_len}" \
      python run_ReTaMTSF_Environment.py \
        --random_seed 2025 \
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
        --devices '0,1,2' \
        --gpu 0 \
        --use_multi_gpu \
        --text_mode "${mode}"
  done
done

echo "Done. Summary saved to ${summary_csv}"
