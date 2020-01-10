for model in ubpr bpr relmf
do
python main.py \
  $model \
  yahoo \
  --threshold 4 \
  --eta 5e-3 \
  --max_iters 301 \
  --batch_size 15 &
done
