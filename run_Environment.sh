export CUDA_VISIBLE_DEVICES=4,2

#for group in "Gasoline Prices|East Coast|New England" "Central Atlantic|Lower Atlantic|Midwest" "Gulf Coast|Rocky Mountain|West Coast"
#  do
#    IFS='|' read -r -a target_ids <<< "$group"   #--target_ids "${target_ids[@]}" \
for target_ids in "Air Quality Index"
do
    for pred_len in 48
    do
      for nref_text in 4 5 6 7 8 9
      do
        python /data/dyl/ReTaMTSF/run_ReTaMTSF_Environment.py \
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
                --target_ids "$target_ids" \
                --batch_size 32 \
                --num_data 1 \
                --patience 10 \
                --train_epochs 100 \
                --nperseg 48 \
                --nref 2 \
                --naggregation 3 \
                --nref_text $nref_text \
                --qt_ref_text $nref_text \
                --seq_len 96 \
                --pred_len $pred_len \
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
                --devices '0,1' \
                --gpu 0 \
                --use_multi_gpu
      done
  done
done
