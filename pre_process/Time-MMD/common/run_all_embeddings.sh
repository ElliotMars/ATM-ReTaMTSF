#!/usr/bin/env bash
set -euo pipefail

# Environment variables
export PATH=~/git-2.41.0/git:$PATH
export PATH=/opt/data/private/ollama/bin:$PATH
export PATH=/opt/conda/bin:$PATH
export OLLAMA_MODELS=/opt/data/private/ollama-models/
export OLLAMA_HOST=0.0.0.0:11434
# Prefer the official hub to avoid mirror rate limits; override if needed.
export HF_ENDPOINT=${HF_ENDPOINT:-https://huggingface.co}

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
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATASETS=("Energy" "Climate" "Health_US" "Environment")
ST_MODELS=("all-MiniLM-L6-v2" "BAAI/bge-small-en-v1.5")

cd "$BASE_DIR"

get_targets() {
  local dataset="$1"
  "$PYTHON" - <<PY
import pandas as pd
path = "../../../dataset/Time-MMD/textual/${dataset}/QueryTextPackage.parquet"
df = pd.read_parquet(path)
for col in df.columns:
    print(col)
PY
}

for dataset in "${DATASETS[@]}"; do
  mapfile -t TARGETS < <(get_targets "$dataset")

  for target in "${TARGETS[@]}"; do
    # Sentence-transformers models
    for model in "${ST_MODELS[@]}"; do
      "$PYTHON" "$BASE_DIR/querytext_preembedding_multi.py" \
        --dataset "$dataset" \
        --target_id "$target" \
        --ForecastingPoint 1 \
        --backend sentence_transformers \
        --model_name "$model"

      "$PYTHON" "$BASE_DIR/querytext_preembedding_multi.py" \
        --dataset "$dataset" \
        --target_id "$target" \
        --ForecastingPoint 0 \
        --backend sentence_transformers \
        --model_name "$model"
    done

  done

  # News embeddings
  for model in "${ST_MODELS[@]}"; do
    "$PYTHON" "$BASE_DIR/newsdatabase_pre_embed_multi.py" \
      --dataset "$dataset" \
      --backend sentence_transformers \
      --model_name "$model"
  done

done
