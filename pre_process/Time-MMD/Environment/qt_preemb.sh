export CUDA_VISIBLE_DEVICES=0
for target in 'Air Quality Index'

do
  python querytext_preembedding.py --target_id "$target" --ForecastingPoint 1
done