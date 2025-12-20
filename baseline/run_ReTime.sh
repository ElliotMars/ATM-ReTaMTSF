# 环境变量相关
export PATH=~/git-2.41.0/git:$PATH
export PATH=/opt/data/private/ollama/bin:$PATH
export PATH=/opt/conda/bin:$PATH
export OLLAMA_MODELS=/opt/data/private/ollama-models/
export OLLAMA_HOST=0.0.0.0:11434
export HF_ENDPOINT=https://hf-mirror.com

# conda相关
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

# 运行 ReTime_real_data.py
MKL_THREADING_LAYER=GNU python ReTime_real_data.py > ./output/program_output 2>&1
